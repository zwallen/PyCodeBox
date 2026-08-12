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
    data : pandas.Series
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

    import pandas as pd

    # Make sure input is pandas series
    data = pd.Series(data)

    # Extract main ICD code groupings if subcategories were given
    if data.str.contains(".").any():
        data = data.str.partition(".")[0]

    # Define ICD to primary site/cancer type dictionary
    primary_site_icd_dict = {
        "Head and Neck": [
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
        "Small Intestine": [
            "C17",  # Small intestine
        ],
        "Colorectal": [
            "C18",  # Colon
            "C19",  # Rectosigmoid junction
            "C20",  # Rectum
        ],
        "Anus/Anal Canal": [
            "C21",  # Anus and anal canal
        ],
        "Liver/Intrahepatic Bile Duct": [
            "C22",  # Liver and intrahepatic bile duct
        ],
        "Gallbladder/Biliary Tract": [
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
        "Thoracic (Non-Lung)": [
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
        "Female Genital": [
            "C51",  # Vulva
            "C52",  # Vagina
            "C57",  # Other female genital organs
            "C58",  # Placenta
        ],
        "Prostate": [
            "C61",  # Prostate
        ],
        "Male Genital": [
            "C60",  # Penis
            "C62",  # Testis
            "C63",  # Other male genital organs
        ],
        "Kidney": [
            "C64",  # Kidney
            "C65",  # Renal pelvis
        ],
        "Ureter/Other Urinary Organs": [
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
        "Adrenal/Other Endocrine": [
            "C74",  # Adrenal gland
            "C75",  # Other endocrine glands
        ],
        "Neuroendocrine": [
            "C7A",  # Neuroendocrine tumor
            "C7B",  # Secondary neuroendocrine tumor
        ],
        "Lymph Node/Lymphoid": [
            "C77",  # Lymph nodes
        ],
        "Secondary/Metastatic": [
            "C78",  # Respiratory/digestive metastases
            "C79",  # Other distant metastases
        ],
        "Unknown Primary": [
            "C80",  # Unknown primary site
        ],
        "Hodgkin Lymphoma": [
            "C81",  # Hodgkin lymphoma
        ],
        "Non-Hodgkin Lymphoma": [
            "C82",  # Follicular lymphoma
            "C83",  # Diffuse NHL
            "C84",  # Mature T/NK-cell lymphoma
            "C85",  # Other NHL
            "C86",  # Other specified T/NK-cell lymphoma
        ],
        "Multiple Myeloma": [
            "C90",  # Multiple myeloma
        ],
        "Leukemia": [
            "C91",  # Lymphoid leukemia
            "C92",  # Myeloid leukemia
            "C93",  # Monocytic leukemia
            "C94",  # Other leukemias
            "C95",  # Unspecified leukemia
        ],
        "Hematopoietic System": [
            "C42",  # Blood, bone marrow, immune/reticuloendothelial systems
        ],
        "Hematopoietic (Other/Undefined)": [
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


def histology_icd_o_3_map(data):
    """
    Map 4-digit ICD-O-3 histology codes to a broad histology group

    This function maps ICD-O-3 histology codes to broad histology groups. It
    first creates an ICD-O-3 code to histology map, then uses the mappings
    to apply histology labels.

    Parameters
    ----------
    data : pandas.Series
        Input list of ICD-O-3 histology codes.

    Returns
    -------
    list[str]
        List of histology group labels.

    Notes
    -----
    * Function only uses the first 4 digits of the ICD-O-3 histology code (i.e.,
      no behavior code).
    * These groups are intended for pan-cancer descriptive analyses.
      They are broad lineage-style buckets, not site-specific disease
      definitions. Site-specific analyses should use more granular
      histology definitions and primary site context where appropriate.
    """

    import pandas as pd

    # Make sure input is pandas series
    data = pd.Series(data)

    # Define ICD-O-3 to histology group dictionary
    histology_icd_o_3_dict = [
        # ------------------------------------------------------------
        # Hematologic malignancies
        # ------------------------------------------------------------
        ((9590, 9729), "Lymphoma"),
        ((9731, 9739), "Plasma Cell Neoplasm"),
        ((9740, 9769), "Other Hematologic/Histiocytic"),
        ((9800, 9949), "Leukemia"),
        ((9950, 9989), "Myeloid/Myelodysplastic/Myeloproliferative"),
        ((9991, 9993), "Myeloid/Myelodysplastic/Myeloproliferative"),
        # ------------------------------------------------------------
        # CNS/neural tumors
        # ------------------------------------------------------------
        ((9350, 9379), "CNS/Neural Tumor"),
        ((9380, 9589), "CNS/Neural Tumor"),
        # ------------------------------------------------------------
        # Germ cell and trophoblastic tumors
        # ------------------------------------------------------------
        ((9060, 9099), "Germ Cell Tumor"),
        ((9100, 9105), "Trophoblastic Tumor"),
        # ------------------------------------------------------------
        # Melanocytic tumors
        # ------------------------------------------------------------
        ((8720, 8799), "Melanoma/Melanocytic"),
        # ------------------------------------------------------------
        # Sarcoma/mesenchymal/mesothelial tumors
        # ------------------------------------------------------------
        ((8800, 8999), "Sarcoma/Mesenchymal"),
        ((9000, 9039), "Sarcoma/Mesenchymal"),
        ((9040, 9045), "Sarcoma/Mesenchymal"),
        ((9050, 9055), "Mesothelioma"),
        ((9110, 9111), "Sarcoma/Mesenchymal"),
        ((9120, 9139), "Sarcoma/Mesenchymal"),
        ((9140, 9140), "Kaposi Sarcoma"),
        ((9150, 9259), "Sarcoma/Mesenchymal"),
        ((9260, 9269), "Ewing/Primitive Neuroectodermal Tumor"),
        # ------------------------------------------------------------
        # Neuroendocrine neoplasms
        # ------------------------------------------------------------
        ((8041, 8045), "Neuroendocrine Neoplasm"),
        ((8150, 8159), "Neuroendocrine Neoplasm"),
        ((8240, 8249), "Neuroendocrine Neoplasm"),
        ((8680, 8719), "Neuroendocrine Neoplasm"),
        # ------------------------------------------------------------
        # Squamous/basal/epidermoid carcinomas
        # ------------------------------------------------------------
        ((8050, 8089), "Squamous Cell Carcinoma"),
        ((8090, 8119), "Basal Cell/Skin Adnexal Carcinoma"),
        ((8120, 8139), "Urothelial/Transitional Cell Carcinoma"),
        # ------------------------------------------------------------
        # Adenocarcinoma and glandular carcinoma spectrum
        # ------------------------------------------------------------
        ((8140, 8149), "Adenocarcinoma"),
        ((8160, 8169), "Biliary/Hepatobiliary Carcinoma"),
        ((8170, 8179), "Hepatocellular Carcinoma"),
        ((8180, 8180), "Biliary/Hepatobiliary Carcinoma"),
        ((8190, 8239), "Adenocarcinoma"),
        ((8250, 8269), "Adenocarcinoma"),
        ((8270, 8289), "Adenocarcinoma"),
        ((8290, 8339), "Endocrine-like/Glandular Carcinoma"),
        ((8340, 8349), "Thyroid Carcinoma"),
        ((8350, 8350), "Thyroid Carcinoma"),
        ((8370, 8379), "Adrenal/Endocrine Carcinoma"),
        ((8380, 8389), "Adenocarcinoma"),
        ((8390, 8429), "Basal Cell/Skin Adnexal Carcinoma"),
        ((8430, 8430), "Non-Adeno/Squamous Carcinoma"),
        ((8440, 8499), "Adenocarcinoma"),
        ((8500, 8549), "Breast/Ductal/Lobular Carcinoma"),
        ((8550, 8579), "Adenocarcinoma"),
        ((8580, 8589), "Thymic Epithelial Tumor"),
        ((8590, 8679), "Gonadal/Sex Cord-Stromal Tumor"),
        # ------------------------------------------------------------
        # Carcinoma NOS/poorly specified epithelial tumors
        # ------------------------------------------------------------
        ((8010, 8039), "Non-Adeno/Squamous Carcinoma"),
        ((8046, 8046), "Non-Adeno/Squamous Carcinoma"),
        # ------------------------------------------------------------
        # Odontogenic and related tumors
        # ------------------------------------------------------------
        ((9270, 9349), "Other/Uncommon Histology"),
        # ------------------------------------------------------------
        # NOS/unclassified malignant neoplasm
        # ------------------------------------------------------------
        ((8000, 8005), "Malignant Neoplasm NOS"),
    ]

    # Define function to perform code matching
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def get_histology_group(code):
        # If code is of type "None"
        if code is None:
            return "Unknown/Unmapped"

        # If code cannot be coerced to an integer
        try:
            code_int = int(str(code).strip())
        except (ValueError, TypeError):
            return "Unknown/Unmapped"

        # If above passes, attempt mapping of code between code ranges
        for (lo, hi), group in histology_icd_o_3_dict:
            if lo <= code_int <= hi:
                return group

        # If all above fail, return
        return "Unknown/Unmapped"

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Perform mapping
    map_results = data.apply(get_histology_group)

    return map_results
