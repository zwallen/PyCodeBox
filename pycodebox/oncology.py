# =============================================================================
# Utility Functions for Oncology Data Processing and Analytics
#
# Description:
# Collection of utility functions for oncology data processing and analysis.
# =============================================================================


def primary_site_icd_map(data):
    """
    Map ICD-10-CM or ICD-O-3 topography codes to a tumor's primary site or
    cancer type

    This function maps ICD-10-CM or SEER ICD-O-3 tumor site prefixes to broad
    primary tumor site groups or cancer types. It first creates an ICD code to
    primary site/cancer type map, turns this map into a lookup table, then
    uses the lookup table to apply primary site/cancer type labels.

    Parameters
    ----------
    data : list[str]
        Input list of ICD-10-CM or ICD-O-3 topography codes.

    Returns
    -------
    list[str]
        List of primary site/cancer type labels.
    
    Notes
    -----
    Function only uses the main ICD code category (digits prior to `.`), and
    subcategories will be ignored if provided.
    """

    # Extract main ICD code groupings if subcategories were given
    if data.str.contains("."):
        data = data.str.partition(".")[0]

    # Define ICD to primary site/cancer type dictionary
    primary_site_icd_dict = {
        "Head and neck": [
            "C00",  # Lip
            "C01",  # Base of tongue
            "C02",  # Tongue
            "C03",  # Gum
            "C04",  # Floor of mouth
            "C05",  # Palate
            "C06",  # Other mouth
            "C07",  # Parotid gland
            "C08",  # Other salivary glands
            "C09",  # Tonsil
            "C10",  # Oropharynx
            "C11",  # Nasopharynx
            "C12",  # Pyriform sinus
            "C13",  # Hypopharynx
            "C14",  # Other pharynx
            "C30",  # Nasal cavity
            "C31",  # Paranasal sinuses
            "C32",  # Larynx
        ],
        "Esophagus": [
            "C15",  # Esophagus
        ],
        "Stomach": [
            "C16",  # Stomach
        ],
        "Small intestine": [
            "C17",  # Small intestine
        ],
        "Colorectal": [
            "C18",  # Colon
            "C19",  # Rectosigmoid junction
            "C20",  # Rectum
        ],
        "Anus/Anal canal": [
            "C21",  # Anus and anal canal
        ],
        "Liver": [
            "C22",  # Liver and intrahepatic bile duct
        ],
        "Gallbladder/Biliary tract": [
            "C23",  # Gallbladder
            "C24",  # Extrahepatic biliary tract
        ],
        "Pancreas": [
            "C25",  # Pancreas
        ],
        "Lung": [
            "C33",  # Trachea
            "C34",  # Bronchus and lung
        ],
        "Thoracic (non-lung)": [
            "C37",  # Thymus
            "C38",  # Heart, mediastinum, pleura
            "C39",  # Other respiratory/intrathoracic sites
        ],
        "Bone": [
            "C40",  # Bones of limbs
            "C41",  # Other bones and joints
        ],
        "Skin": [
            "C43",  # Melanoma of skin
            "C44",  # Non-melanoma skin cancer
            "C4A",  # Merkel cell carcinoma
        ],
        "Mesothelioma": [
            "C45",  # Mesothelioma
        ],
        "Kaposi Sarcoma": [
            "C46",  # Kaposi sarcoma
        ],
        "Soft Tissue/Sarcoma": [
            "C47",  # Peripheral nerves
            "C48",  # Retroperitoneum/peritoneum
            "C49",  # Connective and soft tissue
        ],
        "Breast": [
            "C50",  # Breast
        ],
        "Cervical": [
            "C53",  # Cervix uteri
        ],
        "Endometrial/Uterine": [
            "C54",  # Corpus uteri/endometrium
            "C55",  # Uterus NOS
        ],
        "Ovarian": [
            "C56",  # Ovary
        ],
        "Female genital": [
            "C51",  # Vulva
            "C52",  # Vagina
            "C57",  # Other female genital organs
            "C58",  # Placenta
        ],
        "Prostate": [
            "C61",  # Prostate
        ],
        "Male genital": [
            "C60",  # Penis
            "C62",  # Testis
            "C63",  # Other male genital organs
        ],
        "Kidney": [
            "C64",  # Kidney
            "C65",  # Renal pelvis
        ],
        "Ureter/Other urinary organs": [
            "C66",  # Ureter
            "C68",  # Other urinary organs
        ],
        "Bladder": [
            "C67",  # Bladder
        ],
        "Eye": [
            "C69",  # Eye and adnexa
        ],
        "Brain/CNS": [
            "C70",  # Meninges
            "C71",  # Brain
            "C72",  # Other CNS
        ],
        "Thyroid": [
            "C73",  # Thyroid
        ],
        "Adrenal/Other endocrine": [
            "C74",  # Adrenal gland
            "C75",  # Other endocrine glands
        ],
        "Neuroendocrine": [
            "C7A",  # Neuroendocrine tumor
            "C7B",  # Secondary neuroendocrine tumor
        ],
        "Lymph node/Lymphoid": [
            "C77",  # Lymph nodes
        ],
        "Secondary/Metastatic": [
            "C78",  # Respiratory/digestive metastases
            "C79",  # Other distant metastases
        ],
        "Unknown primary": [
            "C80",  # Unknown primary site
        ],
        "Hodgkin lymphoma": [
            "C81",  # Hodgkin lymphoma
        ],
        "Non-Hodgkin lymphoma": [
            "C82",  # Follicular lymphoma
            "C83",  # Diffuse NHL
            "C84",  # Mature T/NK-cell lymphoma
            "C85",  # Other NHL
            "C86",  # Other specified T/NK-cell lymphoma
        ],
        "Multiple myeloma": [
            "C90",  # Multiple myeloma
        ],
        "Leukemia": [
            "C91",  # Lymphoid leukemia
            "C92",  # Myeloid leukemia
            "C93",  # Monocytic leukemia
            "C94",  # Other leukemias
            "C95",  # Unspecified leukemia
        ],
        "Hematopoietic system": [
            "C42",  # Blood, bone marrow, immune/reticuloendothelial systems
        ],
        "Hematopoietic (other/undefined)": [
            "C88",  # Immunoproliferative disorders
            "C96",  # Other hematopoietic malignancies
        ],
        "Other/Undefined": [
            "C26",  # Other digestive organs
            "C76",  # Ill-defined sites
        ],
    }

    # Create ICD code to site/type lookup table
    primary_site_icd_lookup = {
        code: primary_site
        for primary_site, codes in primary_site_icd_dict.items()
        for code in codes
    }

    # Perform mapping
    map_results = data.map(primary_site_icd_lookup)

    return map_results
