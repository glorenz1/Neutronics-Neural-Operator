import os
import sys
import numpy as np
import openmc
import openmc.model

# reactor parameters taken from Todreas & Kazimi, Nuclear Systems Vol. 1, Appendix K (Seabrook PWR)
PINS_PER_ASM = 17
PIN_PITCH = 1.26
ASM_PITCH = PINS_PER_ASM * PIN_PITCH
FUEL_RADIUS = 0.4096
CLAD_RADIUS = 0.4750

ASM_CONFIGS = [2, 3, 4, 5]
SHAPE_CONFIGS = ["square", "circle", "hexagon"]
SHAPE_IDS = {"square": 0, "circle": 1, "hexagon": 2}
N_CONFIGS = len(ASM_CONFIGS) * len(SHAPE_CONFIGS)

ENRICH_MIN = 0.016
ENRICH_MAX = 0.042
MODE_SETTINGS = {
    "fast": {"n_particles": 1_000_000, "n_batches": 60, "n_inactive": 10},
    "accurate": {"n_particles": 10_000_000, "n_batches": 110, "n_inactive": 10},
}
DEFAULT_MODE = "fast"


# map task_id -> (n_assemblies, shape, get_config)
def get_config(task_id):
    fam = task_id % N_CONFIGS
    return (
        ASM_CONFIGS[fam // len(SHAPE_CONFIGS)],
        SHAPE_CONFIGS[fam % len(SHAPE_CONFIGS)],
        fam,
    )


def sample_enrichment(rng, n_asm):
    total = PINS_PER_ASM * n_asm
    emap = np.zeros((total, total), dtype=np.float32)
    for ai in range(n_asm):
        for aj in range(n_asm):
            base = rng.uniform(ENRICH_MIN, ENRICH_MAX)
            r0 = ai * PINS_PER_ASM
            c0 = aj * PINS_PER_ASM
            pert = rng.normal(0.0, 0.003, size=(PINS_PER_ASM, PINS_PER_ASM))
            emap[r0:r0 + PINS_PER_ASM, c0:c0 + PINS_PER_ASM] = np.clip(
                base + pert, ENRICH_MIN, ENRICH_MAX
            )
    return emap


def make_materials(enrichment_map):
    mats, fuel_map = [], {}

    for enr in np.unique(enrichment_map):
        fuel = openmc.Material(name=f"UO2_{enr:.3f}")
        fuel.set_density("g/cm3", 10.29)
        fuel.add_element("U", 1.0, enrichment=100.0 * enr)
        fuel.add_element("O", 2.0)
        mats.append(fuel)
        fuel_map[float(enr)] = fuel

    water = openmc.Material(name="water")
    water.set_density("g/cm3", 0.7)
    water.add_nuclide("H1", 0.1119, "wo")
    water.add_nuclide("O16", 0.8881, "wo")
    water.add_s_alpha_beta("c_H_in_H2O")
    mats.append(water)

    zircaloy = openmc.Material(name="zircaloy")
    zircaloy.set_density("g/cm3", 6.55)
    zircaloy.add_nuclide("Zr90", 0.5145, "wo")
    zircaloy.add_nuclide("Zr91", 0.1122, "wo")
    zircaloy.add_nuclide("Zr92", 0.1715, "wo")
    zircaloy.add_nuclide("Zr94", 0.1738, "wo")
    zircaloy.add_nuclide("Zr96", 0.0280, "wo")
    mats.append(zircaloy)

    return openmc.Materials(mats), fuel_map, water, zircaloy


def pin_geometry(fuel, water, zircaloy):
    fuel_cyl = openmc.ZCylinder(r=FUEL_RADIUS)
    clad_cyl = openmc.ZCylinder(r=CLAD_RADIUS)
    return openmc.Universe(cells=[
        openmc.Cell(fill=fuel, region=-fuel_cyl),
        openmc.Cell(fill=zircaloy, region=+fuel_cyl & -clad_cyl),
        openmc.Cell(fill=water, region=+clad_cyl),
    ])


def core_geometry(n_asm, shape, enrichment_map, fuel_map, water, zircaloy):
    core_half = n_asm * ASM_PITCH / 2
    total_pins = PINS_PER_ASM * n_asm
    pin_univs = {k: pin_geometry(v, water, zircaloy) for k, v in fuel_map.items()}

    assembly_univs = []
    for ai in range(n_asm):
        row = []
        for aj in range(n_asm):
            r0 = ai * PINS_PER_ASM
            c0 = aj * PINS_PER_ASM
            block = enrichment_map[r0:r0 + PINS_PER_ASM, c0:c0 + PINS_PER_ASM]

            lat = openmc.RectLattice()
            lat.lower_left = [-ASM_PITCH / 2, -ASM_PITCH / 2]
            lat.pitch = [PIN_PITCH, PIN_PITCH]
            lat.universes = [
                [pin_univs[float(block[r, c])] for c in range(PINS_PER_ASM)]
                for r in range(PINS_PER_ASM)
            ]
            row.append(openmc.Universe(cells=[openmc.Cell(fill=lat)]))
        assembly_univs.append(row)

    core_lat = openmc.RectLattice()
    core_lat.lower_left = [-core_half, -core_half]
    core_lat.pitch = [ASM_PITCH, ASM_PITCH]
    core_lat.universes = assembly_univs
    min_z = openmc.ZPlane(-10, boundary_type="reflective")
    max_z = openmc.ZPlane(10, boundary_type="reflective")

    if shape == "square":
        min_x = openmc.XPlane(-core_half, boundary_type="vacuum")
        max_x = openmc.XPlane(core_half, boundary_type="vacuum")
        min_y = openmc.YPlane(-core_half, boundary_type="vacuum")
        max_y = openmc.YPlane(core_half, boundary_type="vacuum")
        region = +min_x & -max_x & +min_y & -max_y

    elif shape == "circle":
        region = -openmc.ZCylinder(r=core_half, boundary_type="vacuum")

    else:  # hexagon
        prism = openmc.model.HexagonalPrism(
            edge_length=core_half,
            orientation="y",
            origin=(0.0, 0.0),
            boundary_type="vacuum",
        )
        region = -prism

    cell = openmc.Cell(fill=core_lat, region=region & +min_z & -max_z)
    return openmc.Geometry(openmc.Universe(cells=[cell])), core_half, total_pins


# create run settings and place a box source over the core volume
def run_settings(seed, core_half, n_particles, n_batches, n_inactive):
    s = openmc.Settings()
    s.seed = int(seed)
    s.batches = n_batches
    s.inactive = n_inactive
    s.particles = n_particles // (n_batches - n_inactive)
    s.run_mode = "eigenvalue"
    bounds = [-core_half, -core_half, -10, core_half, core_half, 10]
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box(bounds[:3], bounds[3:])
    )
    return s


def make_tallies(total_pins, core_half):
    mesh = openmc.RegularMesh()
    mesh.dimension = [total_pins, total_pins, 1]
    mesh.lower_left = [-core_half, -core_half, -10]
    mesh.upper_right = [core_half, core_half, 10]

    tally = openmc.Tally(name="thermal_flux")
    tally.filters = [openmc.MeshFilter(mesh), openmc.EnergyFilter([0.0, 0.625])]
    tally.scores = ["flux"]
    return openmc.Tallies([tally])


# build the geometry mask and coordinate channels used in the dataset
def build_mask(shape, core_half, total_pins):
    dx = 2 * core_half / total_pins
    lin = np.array([-core_half + (i + 0.5) * dx for i in range(total_pins)], dtype=np.float32)
    X, Y = np.meshgrid(lin, lin, indexing="ij")

    if shape == "square":
        mask = np.ones((total_pins, total_pins), dtype=np.float32)
    elif shape == "circle":
        mask = (X**2 + Y**2 <= core_half**2).astype(np.float32)
    else:
        # hexagon matched to HexagonalPrism
        l = core_half
        sqrt3 = np.sqrt(3.0)
        mask = (
            (np.abs(X) <= (sqrt3 / 2.0) * l) &
            (np.abs(Y + X / sqrt3) <= l) &
            (np.abs(Y - X / sqrt3) <= l)
        ).astype(np.float32)
    return (
        mask,
        (X / core_half).astype(np.float32),
        (Y / core_half).astype(np.float32),
        X.astype(np.float32),
        Y.astype(np.float32),
    )


def normalize_flux(raw_flux, mask):
    active = raw_flux * mask
    return active / (np.linalg.norm(active) + 1e-12)


def run_sample(task_id, output_dir, seed, mode):
    out_path = os.path.join(output_dir, f"sample_{task_id:04d}.npz")
    if os.path.exists(out_path):
        print(f"[{task_id:04d}] already exists, skipping.")
        return

    n_asm, shape, family_id = get_config(task_id)
    rng = np.random.default_rng(seed + task_id)
    enrichment_map = sample_enrichment(rng, n_asm)

    materials, fuel_map, water, zircaloy = make_materials(enrichment_map)
    geometry, core_half, total_pins = core_geometry(
        n_asm, shape, enrichment_map, fuel_map, water, zircaloy
    )

    model = openmc.Model(
        geometry=geometry,
        materials=materials,
        settings=run_settings(seed, core_half, **MODE_SETTINGS[mode]),
        tallies=make_tallies(total_pins, core_half),
    )

    run_dir = os.path.join(output_dir, f"run_{task_id}")
    os.makedirs(run_dir, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        threads = int(os.getenv("OMP_NUM_THREADS", "8"))
        model.run(threads=threads, output=False)
        sp = openmc.StatePoint(f"statepoint.{MODE_SETTINGS[mode]['n_batches']}.h5")
        keff = float(sp.keff.nominal_value)
        raw_flux = (
            sp.get_tally(name="thermal_flux")
            .get_reshaped_data(expand_dims=True)
            .reshape(total_pins, total_pins)
            .astype(np.float32)
        )
    finally:
        os.chdir(cwd)

    mask, coord_xn, coord_yn, coord_xcm, coord_ycm = build_mask(
        shape, core_half, total_pins
    )
    flux = normalize_flux(raw_flux, mask)
    enrich = enrichment_map * mask

    np.savez(
        out_path,
        enrich=enrich,
        flux=flux,
        mask=mask,
        coord_xn=coord_xn,
        coord_yn=coord_yn,
        coord_xcm=coord_xcm,
        coord_ycm=coord_ycm,
        keff=np.float32(keff),
        n_assemblies=np.int8(n_asm),
        shape_id=np.int8(SHAPE_IDS[shape]),
        family_id=np.int8(family_id),
        core_half_cm=np.float32(core_half),
        dx_cm=np.float32(PIN_PITCH),
    )

    print(
        f"[{task_id:04d}] fam={family_id:02d} n_asm={n_asm} shape={shape:8s} "
        f"grid={total_pins}² k_eff={keff:.5f} mode={mode} → {out_path}"
    )


# launch a single sample job for slurm scheduler
def main():
    args = sys.argv[1:]
    if len(args) < 3:
        print("Usage:")
        print("python openmc_data.py <task_id> <output_dir> <seed> [fast|accurate]")
        sys.exit(1)

    task_id = int(args[0])
    output_dir = args[1]
    seed = int(args[2])
    mode = args[3] if len(args) >= 4 else DEFAULT_MODE

    if mode not in MODE_SETTINGS:
        sys.exit(f'Unknown mode "{mode}". Choose: {list(MODE_SETTINGS)}')

    os.makedirs(output_dir, exist_ok=True)
    run_sample(task_id, output_dir, seed, mode)


if __name__ == "__main__":
    main()
