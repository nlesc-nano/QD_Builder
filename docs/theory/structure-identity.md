(theory-identity)=
# Structure identity and size

A library of nanocrystal models needs answers to three questions that
single builds never raise: how large is a particle, where is its centre,
and when are two structures the same? The answers must be invariant to
rotation and translation, cheap to compute for thousands of atoms, and
meaningful for both freshly built and relaxed geometries. The functions in
`builder.library_record` provide them. They are shared by the library
generator and by the QDSpace ingest.

## The inorganic core

All size and centre descriptors refer to the *native* atoms, i.e. the
elements of the bulk material (`native_elements`), and ignore ligands.
Ligands are chemically replaceable and their number varies with the
passivation strategy, while the inorganic framework defines the particle.
In this sense "core" means the native part of a particle, which is distinct
from the core region of a core/shell structure.

## Size

The primary size descriptor is an equivalent-sphere diameter derived from
the radius of gyration. For $N$ core atoms at positions $\mathbf r_i$ with
centroid $\mathbf c$,

$$
R_g = \Bigl(\tfrac1N \sum_i \lvert\mathbf r_i - \mathbf c\rvert^2\Bigr)^{1/2}.
$$

For a homogeneous sphere of radius $R$ one has $R_g^2 = \tfrac35 R^2$.
The diameter of the sphere with the same radius of gyration is therefore

$$
d = 2\sqrt{5/3}\;R_g .
$$

This diameter is robust against individual protruding atoms and is the
value used by the library's size filter. Two further descriptors are
recorded: the maximum diameter $2\max_i\lvert\mathbf r_i - \mathbf c\rvert$,
and the diameter of the sphere with the volume of the convex hull,
$2\,(3V_\text{hull}/4\pi)^{1/3}$.

## Centre

The centre of a particle is characterised by the atom closest to the
centroid of the core. If a native atom lies within 0.5 Å of the centroid,
the particle is labelled by that atom's species: *Cd-centred*,
*Se-centred*, and so on. Otherwise it is labelled *interstitial*. The
centre matters physically because a cation-centred and an anion-centred
particle of similar size have different symmetries and stoichiometries.
The library generator rejects any build whose detected centre differs from
the species it was asked to centre on.

## Exact identity: the fingerprint

Two structures are considered identical when their species-resolved radial
distributions coincide. For every species the distances of all its atoms
to the centroid of the whole structure are sorted and binned at 0.05 Å.
The resulting lists, ordered by species, are hashed (SHA-1, first 16 hex
digits). The fingerprint is invariant to rotation, translation and atom
order, and it separates distinct shapes and stoichiometries reliably. The
library generator uses it to discard duplicates, keeping each structure at
the smallest size that produced it.

## Tolerant identity: the radial signature

A relaxed structure has the same topology as the model it started from,
but its distances have shifted, so the fingerprints differ. For such
comparisons the unbinned per-species distance lists form a *radial
signature*. The distance between two signatures is infinite if their
species or per-species counts differ. Otherwise it is the mean absolute
difference of corresponding sorted distances, in ångström. A small value
(QDSpace uses 0.5 Å) identifies a relaxed DFT structure as the *twin* of a
builder structure.

## Structure identifiers

Each record receives the identifier `material-centre-formula-surface`, for
example `CdSe-Se-Cd68Se55Cl26-clean`. The formula lists the native elements
first, in the order given by the configuration, followed by the ligands in
alphabetical order.
