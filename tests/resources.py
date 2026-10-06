"""The models of the tests, copied from the resources of sbmlutils."""

from pathlib import Path

#: the directory of the models
MODELS_DIR: Path = Path(__file__).parent / "data" / "models"

DEMO_SBML: Path = MODELS_DIR / "demo" / "Koenig_demo_v15.xml"
REPRESSILATOR_SBML: Path = MODELS_DIR / "repressilator" / "BIOMD0000000012_urn.xml"
COMP_ICG_BODY: Path = MODELS_DIR / "comp" / "icg_body.xml"
COMP_ICG_BODY_FLAT: Path = MODELS_DIR / "comp" / "icg_body_flat.xml"
GALACTOSE_SINGLECELL_SBML: Path = MODELS_DIR / "galactose" / "galactose_30.xml"
VDP_SBML: Path = MODELS_DIR / "van_der_pol" / "van_der_pol.xml"
COMP_DEX_LIVER: Path = MODELS_DIR / "comp" / "dex_liver.xml"
COMP_SPT_LIVER: Path = MODELS_DIR / "comp" / "spt_liver.xml"
