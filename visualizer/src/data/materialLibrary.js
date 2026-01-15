export const seedMaterials = [
  {
    id: "titanium-alloy-alpha",
    name: "Alpha Titanium Alloy",
    composition: { ti: 0.92, al: 0.05, v: 0.03 },
    baseLattice: "hcp",
    stability: 0.78,
    descriptors: { density: 4.43, hardness: 3.1, entropy: 0.42 },
  },
  {
    id: "perovskite-oxide",
    name: "Perovskite Oxide",
    composition: { sr: 0.22, ti: 0.26, o: 0.52 },
    baseLattice: "perovskite",
    stability: 0.68,
    descriptors: { density: 5.11, hardness: 2.9, entropy: 0.58 },
  },
  {
    id: "2d-hex-boron-nitride",
    name: "Hexagonal Boron Nitride",
    composition: { b: 0.49, n: 0.51 },
    baseLattice: "hexagonal",
    stability: 0.83,
    descriptors: { density: 2.1, hardness: 1.8, entropy: 0.33 },
  },
  {
    id: "high-entropy-alloy",
    name: "High Entropy Alloy",
    composition: { fe: 0.2, cr: 0.2, ni: 0.2, co: 0.2, mn: 0.2 },
    baseLattice: "fcc",
    stability: 0.71,
    descriptors: { density: 7.8, hardness: 4.2, entropy: 0.91 },
  },
  {
    id: "meta-lattice-aerogel",
    name: "Meta Lattice Aerogel",
    composition: { c: 0.32, si: 0.22, o: 0.46 },
    baseLattice: "diamond",
    stability: 0.54,
    descriptors: { density: 0.9, hardness: 1.1, entropy: 0.73 },
  },
];

export function byId(targetId) {
  return seedMaterials.find((material) => material.id === targetId);
}
