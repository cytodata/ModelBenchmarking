"""Look up a compound's common name and mechanism of action online.

Queries the public PubChem and ChEMBL REST APIs, so an internet connection is
required. Results are cached in memory for the lifetime of the Python session.
"""

import logging
import re

import requests

ANNOTATION_CACHE: dict[str, tuple[str, str]] = {}


def get_rich_compound_annotation(inchikey: str) -> tuple[str, str]:
    """Return a readable name and mechanism of action for an InChIKey.

    Three sources are tried in order:

    1. **PubChem synonyms** for a human-readable name (skipping CAS numbers
       and vendor/database identifiers).
    2. **ChEMBL curated mechanism of action**. If the exact InChIKey is not in
       ChEMBL, the first 14 characters (the molecular skeleton, ignoring
       stereochemistry and protonation) are used instead.
    3. **ChEMBL bioactivity fallback**: the target of one IC50 record with a
       value <= 500 (reported as "<target> inhibitor").

    Parameters
    ----------
    inchikey : str

    Returns
    -------
    name : str
        ``"Unknown"`` if nothing was found.
    moa : str
        ``"Unannotated"`` if nothing was found.

    Notes
    -----
    * Network and response-parsing errors are logged as warnings and yield
      "Unknown"/"Unannotated".
    * The skeleton-level match in step 2 can return a stereoisomer or salt of
      a different registered compound.
    * The IC50 filter in step 3 does not restrict ``standard_units``, so the
      500 cutoff is not guaranteed to be in nM, and "inhibitor" is assumed.
    """
    if inchikey in ANNOTATION_CACHE:
        return ANNOTATION_CACHE[inchikey]

    name = "Unknown"
    moa = "Unannotated"

    # --- TIER 1: PubChem Synonyms ---
    try:
        url_syn = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/synonyms/JSON"
        r = requests.get(url_syn, timeout=4)
        if r.status_code == 200:
            info = r.json().get("InformationList", {}).get("Information", [{}])[0]
            synonyms = info.get("Synonym", [])
            for s in synonyms[:15]:
                if (
                    not re.match(r"^\d+-\d+-\d+$", s)
                    and len(s) > 2
                    and not any(s.startswith(prefix) for prefix in ["SCHEMBL", "CHEMBL", "AKOS", "CS-", "EN300"])
                ):
                    name = s
                    break
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as exc:
        logging.warning("PubChem annotation lookup failed: %s", exc)

    # --- TIER 2: ChEMBL with Part II Block 1 Rescue ---
    chembl_id = None
    try:
        url_chembl = (
            f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_structures__standard_inchi_key={inchikey}"
        )
        r = requests.get(url_chembl, timeout=4)
        mols = r.json().get("molecules", []) if r.status_code == 200 else []

        if not mols and len(inchikey) >= 14:
            b1 = inchikey[:14]
            url_b1 = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_structures__standard_inchi_key__startswith={b1}"
            r = requests.get(url_b1, timeout=4)
            mols = r.json().get("molecules", []) if r.status_code == 200 else []

        if mols:
            mol = mols[0]
            chembl_id = mol.get("molecule_chembl_id")
            if name == "Unknown" and mol.get("pref_name"):
                name = mol.get("pref_name")

            url_mech = f"https://www.ebi.ac.uk/chembl/api/data/mechanism.json?molecule_chembl_id={chembl_id}"
            r_m = requests.get(url_mech, timeout=4)
            if r_m.status_code == 200:
                mechs = r_m.json().get("mechanisms", [])
                if mechs:
                    moa = mechs[0].get("mechanism_of_action")

            # --- TIER 3: ChEMBL Bioactivity Target Fallback (< 500nM) ---
            if moa == "Unannotated" and chembl_id:
                url_act = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id={chembl_id}&standard_type=IC50&standard_value__lte=500&limit=1"
                r_a = requests.get(url_act, timeout=4)
                if r_a.status_code == 200:
                    acts = r_a.json().get("activities", [])
                    if acts and acts[0].get("target_pref_name"):
                        target = acts[0].get("target_pref_name")
                        moa = f"{target} inhibitor (IC50 < 500nM)"
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as exc:
        logging.warning("ChEMBL annotation lookup failed: %s", exc)

    ANNOTATION_CACHE[inchikey] = (name, moa)
    return name, moa
