(post-alloying)=
# Alloying

## Substitution on the lattice

Alloyed and doped nanocrystals, such as Cd₁₋ₓZnₓSe, CdSe₁₋ₓSₓ or
Mn-doped particles, are made by incorporating a second cation or anion into
the lattice of the host. In the builder, alloying is a substitution. A
fraction of the atoms of one native species is relabelled as another
element, on the positions of the host lattice. No relaxation is performed,
so the local strain caused by different ionic radii is left to a subsequent
geometry optimisation.

The substitution can be restricted to a region of the particle. With
`region: surface`, only atoms in the surface shell are replaced, which
models surface enrichment or cation exchange that has not yet penetrated
the particle. With `region: core`, only interior atoms are replaced. The
default, `both`, samples the whole particle. The number of substituted
atoms is the requested fraction or count of the eligible atoms, and the
spatial distribution follows the conventions of {ref}`post-overview`. The
`uniform` mode gives a dilute, evenly dispersed alloy, and `segmented`
gives a clustered, phase-segregated one.

## Charge

An isovalent substitution, such as Cd²⁺ by Zn²⁺ or Se²⁻ by S²⁻, leaves the
charge unchanged. An aliovalent one, such as In³⁺ in a II–VI lattice,
changes it by $(q_\text{new} - q_\text{old})$ per substituted atom. After
alloying the builder therefore runs the full charge balance of
{ref}`theory-charge-passivation` again, with the same options as the
initial passivation. This may add or remove ligand placeholders or surface
ions to restore neutrality. The charge of the new element is taken from
`with_charge` or, if it is already listed, from the `charges` block, which
takes precedence.

## Parameters

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | run the treatment |
| `seed` | 1337 | random seed |
| `passes[].replace` | required | native element to substitute |
| `passes[].with` | required | substituting element |
| `passes[].with_charge` | from `charges` | its formal charge |
| `passes[].region` | `both` | `surface`, `core` or `both` |
| `passes[].ratio` / `target_count` | 1.0 / 0 | fraction or number of atoms |
| `passes[].distribution` | `random` | `random`, `uniform` or `segmented` |

The manifest records an `alloying_ledger` with the substitution, region,
count and charge change of every pass.

## Example

Replacing 20 % of the surface Cd of a CdSe particle by Zn, spread
uniformly. This recipe is illustrative:

```yaml
charges: {Cd: 2, Se: -2, Cl: -1, Zn: 2}
post_treatment:
  alloying:
    enabled: true
    passes:
      - replace: Cd
        with: Zn
        region: surface
        distribution: uniform
        ratio: 0.2
```
