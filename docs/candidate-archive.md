# Candidate Archive

## Candidate 1: Internal Site Construction (Topos Theory)

**Source:** Session 1, Topos Theory investigation. Primary: Mac Lane & Moerdijk, Johnstone "Sketches of an Elephant" Part C.

**FORMAL STRUCTURE EXTRACTION:**
- Old format: Objects and logic within a fixed topos E
- New format: Objects and logic within Sh_E(C, J), a bounded topos over E with different internal logic
- Transition op: Construct internal site (C, J) in E; form sheaf topos Sh_E(C, J) → E
- Became expressible: Non-Boolean logic, local-to-global constructions, cohomology, different truth values
- Computable: YES for decidable internal sites

**CAGE DIAGNOSIS:**
- Layer affected: FORMAT_CHANGE (the logic itself changes)
- Within-format: NO — the internal language of Sh_E(C,J) differs from E's
- Self-applicable: PARTIALLY — the system can construct internal sites and Lawvere-Tierney topologies, but cannot escape the ambient topos entirely

**BLACKLIST CHECK:** CLEAR. Not B01 (site is constructed, not hardcoded). Not B05 (bounded only in the sense of being over E, but the new topos has genuinely new logic). Not B06 (sites are not templates). Not B14 (metaspace changes with each site choice).

**TRANSPLANT SKETCH:**
A computational system would need:
1. A representation of its own "category of computations" as an internal category
2. The ability to construct new Grothendieck topologies on that category
3. A sheafification procedure that produces a new computational framework
4. Migration of existing computations into the new framework via inverse image

**KEY DIFFICULTY:** Requires the system to represent its own computation category, which is a form of reflection. The bounded-topos result says this is achievable but limited — you get new topoi OVER your current one, not arbitrary external ones.

**VERDICT:** STRUCTURAL_EXPANSION. Strongest candidate from Session 1.

---

## Candidate 2: Yoneda Embedding (Category Theory)

**Source:** Session 1, Category Theory investigation. Primary: Mac Lane Ch. III.

**FORMAL STRUCTURE EXTRACTION:**
- Old format: Object c in category C (possibly lacking limits, colimits, exponentials)
- New format: Representable presheaf y(c) = Hom(-, c) in [C^op, Set] (a topos with all limits, colimits, subobject classifier, exponentials)
- Transition op: Yoneda embedding y: C → [C^op, Set], full and faithful
- Became expressible: All small limits/colimits, exponential objects, subobject classifier, higher-order intuitionistic logic, "generalized objects" (non-representable presheaves)
- Computable: YES for finite categories

**CAGE DIAGNOSIS:**
- Layer affected: FORMAT_CHANGE (category gains topos structure)
- Within-format: NO — presheaf category is strictly richer than C
- Self-applicable: The embedding is canonical (no choices), but applying it to one's own computation category requires reflection

**BLACKLIST CHECK:** CLEAR.

**TRANSPLANT SKETCH:**
1. Represent the system's current operations as a finite category C
2. Compute the presheaf category [C^op, FinSet]
3. The result is a topos — gains exponentials (higher-order functions), all limits/colimits
4. Non-representable presheaves are "ideal elements" — computations that don't exist in C but are well-defined as limits of things that do

**KEY DIFFICULTY:** For finite C, this is fully computable. But the presheaf category is exponentially larger than C. The density theorem (every presheaf is a colimit of representables) provides a basis, but the colimit computations may be expensive.

**VERDICT:** STRUCTURAL_EXPANSION. Most computationally concrete candidate.

---

## Candidate 3: Realizability Topos (Topos Theory)

**Source:** Session 1, Topos Theory investigation. Primary: Hyland "The Effective Topos" (1982), van Oosten "Realizability."

**FORMAL STRUCTURE EXTRACTION:**
- Old format: Set-theoretic existence (arbitrary functions N→N, classical logic, LEM)
- New format: Computationally realized existence (every function N→N is computable, CT_0 holds, Markov's principle, LEM fails)
- Transition op: Given PCA A=(A,·), construct realizability topos RT(A) via tripos-to-topos construction
- Became expressible: Internal Church-Turing thesis, uniform continuity of all [0,1]→R functions, computable existence as default
- Computable: YES (morphisms tracked by recursive functions)

**CAGE DIAGNOSIS:**
- Layer affected: FORMAT_CHANGE (lateral shift — different things expressible, not strictly more)
- Within-format: NO — Eff is neither subcategory nor supercategory of Set
- Self-applicable: Partially — a PCA can construct sub-PCAs, generating sub-realizability topoi

**BLACKLIST CHECK:** CLEAR. Not B02 (not a stack machine — full topos structure). Not B04 (not crystallization). Not B14 (PCA is a parameter).

**TRANSPLANT SKETCH:**
1. Define the system's computational operations as a PCA
2. Construct the realizability topos over this PCA
3. In this topos, all internal functions are "tracked" — every operation has a realizer
4. The internal CT_0 means the system can reason about computability internally

**KEY DIFFICULTY:** The realizability topos is a lateral shift, not a strict expansion. Some set-theoretic constructions become inexpressible (non-computable functions). The question is whether the lateral shift provides what RSI needs — access to new computations — or merely restricts to computable ones (which a TM already has).

**VERDICT:** STRUCTURAL_EXPANSION (lateral). Needs further analysis in Session 9 for RSI applicability.
