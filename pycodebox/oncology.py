# =============================================================================
# Utility Functions for Oncology Data Processing and Analytics
#
# Description:
# Collection of utility functions for oncology data processing and analysis.
# =============================================================================


def primary_site_icd_map(data, include_d_codes=False):
    """
    Map ICD-10-CM or ICD-O-3 topography codes to a tumor's primary site or
    cancer type

    This function maps ICD-10-CM or SEER ICD-O-3 tumor site prefixes to broad
    primary tumor site groups or cancer types. It first creates an ICD code to
    primary site/cancer type map, turns this map into a lookup table, then
    uses the lookup table to apply primary site/cancer type labels.

    Parameters
    ----------
    data : array-like
        ICD-10-CM or ICD-O-3 topography codes. Codes may contain or omit
        decimal points.
    include_d_codes : bool, default: False
        Whether to include ICD-10-CM D-codes for carcinoma in situ or
        neoplasms of unspecified/uncertain in mapping. D-codes related to
        benign neoplasms are not implemented.

    Returns
    -------
    pandas.Series
        Broad primary-site or cancer-type labels aligned to the input index.
        Unsupported or invalid codes are returned as missing values.

    Notes
    -----
    ICD-10-CM C codes and ICD-O-3 topography codes are mapped using their
    three-character category, including alphanumeric categories such as
    C4A, C7A, and C7B.

    Selected ICD-10-CM D codes are mapped using the shortest site-specific
    prefix that can be assigned unambiguously to a broad cancer group.
    Prefix lengths vary because some D-code categories contain descendants
    assigned to different anatomical groups. Decimal points, surrounding
    whitespace, and letter case are ignored.
    """

    import pandas as pd
    import re

    # Standardize input
    data = pd.Series(data, dtype="string")
    data = data.str.strip().str.upper().str.replace(".", "", regex=False)

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
            "C30",  # Nasal cavity/middle ear
            "C31",  # Accessory sinuses
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
        "Hepatobiliary": [
            "C22",  # Liver and intrahepatic bile duct
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
        "Melanoma": [
            "C43",  # Melanoma of skin
        ],
        "Non-Melanoma Skin": [
            "C44",  # Non-melanoma skin cancer
        ],
        "Merkel Cell": [
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
            "C49",  # Connective and soft tissue
        ],
        "Breast": [
            "C50",  # Breast
        ],
        "Female Genital": [
            "C51",  # Vulva
            "C52",  # Vagina
            "C57",  # Other female genital organs
            "C58",  # Placenta
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
        "Placenta/Gestational Trophoblastic": [
            "C58",  # Placenta
        ],
        "Male Genital": [
            "C60",  # Penis
            "C62",  # Testis
            "C63",  # Other male genital organs
        ],
        "Prostate": [
            "C61",  # Prostate
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
        "Endocrine": [
            "C73",  # Thyroid
            "C74",  # Adrenal gland
            "C75",  # Other endocrine glands
        ],
        "Neuroendocrine": [
            "C7A",  # Neuroendocrine tumor
        ],
        "Secondary/Metastatic": [
            "C77",  # Secondary/unspecified lymph nodes
            "C78",  # Respiratory/digestive metastases
            "C79",  # Other distant metastases
            "C7B",  # Secondary neuroendocrine tumor
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
        "Multiple Myeloma/Plasma Cell Neoplasms": [
            "C90",  # Multiple myeloma and plasma cell neoplasms
        ],
        "Leukemia": [
            "C91",  # Lymphoid leukemia
            "C92",  # Myeloid leukemia
            "C93",  # Monocytic leukemia
            "C94",  # Other leukemias
            "C95",  # Unspecified leukemia
        ],
        "Hematopoietic (Other/Undefined)": [
            "C88",  # Immunoproliferative disorders
            "C96",  # Other hematopoietic malignancies
        ],
        "Other/Undefined": [
            "C26",  # Other digestive organs
            "C48",  # Retroperitoneum/peritoneum
            "C76",  # Ill-defined sites
        ],
        ## ICD-O-3 topography only
        "Hematopoietic System": [
            "C42",  # Blood, bone marrow, immune/reticuloendothelial systems
        ],
    }

    # Add ICD-10-CM D-codes if requested
    if include_d_codes:
        primary_site_icd_dict["Head and Neck"].extend(
            [
                "D000",  # Carcinoma in situ of lip, oral cavity and pharynx
                "D020",  # Carcinoma in situ of larynx
                "D370",  # Neoplasm of uncertain behavior of lip, oral cavity and pharynx
                "D380",  # Neoplasm of uncertain behavior of larynx
            ]
        )
        primary_site_icd_dict["Esophagus"].extend(
            [
                "D001",  # Carcinoma in situ of esophagus
            ]
        )
        primary_site_icd_dict["Stomach"].extend(
            [
                "D002",  # Carcinoma in situ of stomach
                "D371",  # Neoplasm of uncertain behavior of stomach
            ]
        )
        primary_site_icd_dict["Small Intestine"].extend(
            [
                "D372",  # Neoplasm of uncertain behavior of small intestine
            ]
        )
        primary_site_icd_dict["Colorectal"].extend(
            [
                "D010",  # Carcinoma in situ of colon
                "D011",  # Carcinoma in situ of rectosigmoid junction
                "D012",  # Carcinoma in situ of rectum
                "D014",  # Carcinoma in situ of other/unspecified part of intestine
                "D373",  # Neoplasm of uncertain behavior of appendix
                "D374",  # Neoplasm of uncertain behavior of colon
                "D375",  # Neoplasm of uncertain behavior of rectum
            ]
        )
        primary_site_icd_dict["Anus/Anal Canal"].extend(
            [
                "D013",  # Carcinoma in situ of anus and anal canal
            ]
        )
        primary_site_icd_dict["Hepatobiliary"].extend(
            [
                "D015",  # Carcinoma in situ of liver, gallbladder and bile ducts
                "D376",  # Neoplasm of uncertain behavior of liver, gallbladder and bile ducts
            ]
        )
        primary_site_icd_dict["Lung"].extend(
            [
                "D021",  # Carcinoma in situ of trachea
                "D022",  # Carcinoma in situ of lung
                "D381",  # Neoplasm of uncertain behavior of trachea, bronchus and lung
            ]
        )
        primary_site_icd_dict["Thoracic (Non-Lung)"].extend(
            [
                "D023",  # Carcinoma in situ of other parts of respiratory system
                "D382",  # Neoplasm of uncertain behavior of pleura
                "D383",  # Neoplasm of uncertain behavior of mediastinum
                "D384",  # Neoplasm of uncertain behavior of thymus
                "D385",  # Neoplasm of uncertain behavior of other respiratory organs
            ]
        )
        primary_site_icd_dict["Bone"].extend(
            [
                "D480",  # Neoplasm of uncertain behavior of bone and articular cartilage
            ]
        )
        primary_site_icd_dict["Melanoma"].extend(
            [
                "D03",  # Melanoma in situ
            ]
        )
        primary_site_icd_dict["Non-Melanoma Skin"].extend(
            [
                "D04",  # Carcinoma in situ of skin
                "D485",  # Neoplasm of uncertain behavior of skin
            ]
        )
        primary_site_icd_dict["Soft Tissue/Sarcoma"].extend(
            [
                "D481",  # Other specified neoplasm of uncertain behavior of connective and other soft tissue
                "D482",  # Neoplasm of uncertain behavior of peripheral nerves and autonomic nervous system
            ]
        )
        primary_site_icd_dict["Breast"].extend(
            [
                "D05",  # Carcinoma in situ of breast
                "D486",  # Neoplasm of uncertain behavior of breast
                "D493",  # Neoplasm of unspecified behavior of breast
            ]
        )
        primary_site_icd_dict["Female Genital"].extend(
            [
                "D071",  # Carcinoma in situ of vulva
                "D072",  # Carcinoma in situ of vagina
                "D073",  # Carcinoma in situ of other/unspecified female genital organs
                "D398",  # Neoplasm of uncertain behavior of other specified female genital organs
                "D399",  # Neoplasm of uncertain behavior of female genital organ, unspecified
            ]
        )
        primary_site_icd_dict["Cervical"].extend(
            [
                "D06",  # Carcinoma in situ of cervix
            ]
        )
        primary_site_icd_dict["Endometrial/Uterine"].extend(
            [
                "D070",  # Carcinoma in situ of endometrium
                "D390",  # Neoplasm of uncertain behavior of uterus
            ]
        )
        primary_site_icd_dict["Ovarian"].extend(
            [
                "D391",  # Neoplasm of uncertain behavior of ovary
            ]
        )
        primary_site_icd_dict["Placenta/Gestational Trophoblastic"].extend(
            [
                "D392",  # Neoplasm of uncertain behavior of placenta
            ]
        )
        primary_site_icd_dict["Male Genital"].extend(
            [
                "D074",  # Carcinoma in situ of penis
                "D076",  # Carcinoma in situ of other/unspecified male genital organs
                "D401",  # Neoplasm of uncertain behavior of testis
                "D408",  # Neoplasm of uncertain behavior of other specified male genital organs
                "D409",  # Neoplasm of uncertain behavior of male genital organ, unspecified
            ]
        )
        primary_site_icd_dict["Prostate"].extend(
            [
                "D075",  # Carcinoma in situ of prostate
                "D400",  # Neoplasm of uncertain behavior of prostate
            ]
        )
        primary_site_icd_dict["Kidney"].extend(
            [
                "D4951",  # Neoplasm of unspecified behavior of kidney
                "D410",  # Neoplasm of uncertain behavior of kidney
            ]
        )
        primary_site_icd_dict["Ureter/Other Urinary Organs"].extend(
            [
                "D091",  # Carcinoma in situ of other/unspecified urinary organ
                "D411",  # Neoplasm of uncertain behavior of renal pelvis
                "D412",  # Neoplasm of uncertain behavior of ureter
                "D413",  # Neoplasm of uncertain behavior of urethra
                "D418",  # Neoplasm of uncertain behavior of other specified urinary organs
                "D419",  # Neoplasm of uncertain behavior of unspecified urinary organ
            ]
        )
        primary_site_icd_dict["Bladder"].extend(
            [
                "D090",  # Carcinoma in situ of bladder
                "D414",  # Neoplasm of uncertain behavior of bladder
                "D494",  # Neoplasm of unspecified behavior of bladder
            ]
        )
        primary_site_icd_dict["Eye"].extend(
            [
                "D092",  # Carcinoma in situ of eye
                "D4981",  # Neoplasm of unspecified behavior of retina and choroid
            ]
        )
        primary_site_icd_dict["Brain/CNS"].extend(
            [
                "D42",  # Neoplasm of uncertain behavior of the meninges
                "D43",  # Neoplasm of uncertain behavior of brain and central nervous system
                "D496",  # Neoplasm of unspecified behavior of brain
            ]
        )
        primary_site_icd_dict["Endocrine"].extend(
            [
                "D093",  # Carcinoma in situ of thyroid and other endocrine glands
                "D44",  # Neoplasm of uncertain behavior of endocrine glands
            ]
        )
        primary_site_icd_dict["Hematopoietic (Other/Undefined)"].extend(
            [
                "D45",  # Polycythemia vera
                "D46",  # Myelodysplastic syndromes
                "D47",  # Other neoplasms of uncertain behavior of lymphoid, hematopoietic and related tissue
            ]
        )
        primary_site_icd_dict["Other/Undefined"].extend(
            [
                "D017",  # Carcinoma in situ of other specified digestive organs
                "D019",  # Carcinoma in situ of digestive organ, unspecified
                "D024",  # Carcinoma in situ of respiratory system, unspecified
                "D098",  # Carcinoma in situ of other specified sites
                "D099",  # Carcinoma in situ, unspecified
                "D378",  # Neoplasm of uncertain behavior of other specified digestive organs
                "D379",  # Neoplasm of uncertain behavior of digestive organ, unspecified
                "D386",  # Neoplasm of uncertain behavior of respiratory organ, unspecified
                "D483",  # Neoplasm of uncertain behavior of retroperitoneum
                "D484",  # Neoplasm of uncertain behavior of peritoneum
                "D487",  # Neoplasm of uncertain behavior of other specified sites
                "D489",  # Neoplasm of uncertain behavior, unspecified
                "D490",  # Neoplasm of unspecified behavior of digestive system
                "D491",  # Neoplasm of unspecified behavior of respiratory system
                "D492",  # Neoplasm of unspecified behavior of bone, soft tissue, and skin
                "D4959",  # Neoplasm of unspecified behavior of other genitourinary organ
                "D497",  # Neoplasm of unspecified behavior of endocrine glands and other parts of nervous system
                "D4989",  # Neoplasm of unspecified behavior of other specified sites
                "D499",  # Neoplasm of unspecified behavior of unspecified site
            ]
        )

    # Reverse dictionary to create ICD code-to-site/type lookup
    primary_site_icd_lookup = {
        code: primary_site
        for primary_site, codes in primary_site_icd_dict.items()
        for code in codes
    }

    # Confirm that no prefix is assigned to multiple broad cancer types
    lookup_rows = [
        (code, primary_site)
        for primary_site, codes in primary_site_icd_dict.items()
        for code in codes
    ]
    lookup_df = pd.DataFrame(
        lookup_rows,
        columns=["CodePrefix", "BroadCancerType"],
    )
    conflicts = (
        lookup_df.groupby("CodePrefix")["BroadCancerType"]
        .nunique()
        .loc[lambda x: x > 1]
    )
    if not conflicts.empty:
        raise ValueError(
            "ICD code prefixes assigned to multiple broad cancer types: "
            f"{conflicts.index.tolist()}"
        )

    # Match more-specific prefixes before shorter prefixes
    code_prefixes = sorted(
        primary_site_icd_lookup,
        key=len,
        reverse=True,
    )

    # Anchor matches at the beginning of the normalized code
    pattern = re.compile(
        r"^(" + "|".join(re.escape(code) for code in code_prefixes) + r")"
    )

    # Define function to perform code matching
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def extract_site(x):
        if pd.isna(x):
            return pd.NA
        match = pattern.match(x)
        if match:
            return primary_site_icd_lookup[match.group(0)]
        return pd.NA

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Perform mapping
    map_results = data.map(extract_site).astype("string")

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
    pandas.Series
        List of histology group labels.

    Notes
    -----
    Function only uses the first 4 digits of the ICD-O-3 histology code (i.e.,
    no behavior code).

    These groups are intended for pan-cancer descriptive analyses. They are
    broad lineage-style buckets, not site-specific disease definitions.
    Site-specific analyses should use more granular histology definitions
    and primary site context where appropriate.
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
        ((8160, 8169), "Cholangiocarcinoma"),
        ((8170, 8179), "Hepatocellular Carcinoma"),
        ((8180, 8180), "Cholangiocarcinoma"),
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


def histology_regex_map(data):
    """
    Create broad histology groups using regex pattern matching

    This function uses regular expression pattern matching to map specific
    histologies to broad histology groups. Histology groups are derived from
    free‑text diagnoses using a rule‑based mapper aligned to TCGA and AACR GENIE
    lineage conventions. Ambiguous or mixed histologies (e.g., carcinosarcoma,
    NOS neoplasms) are collapsed into ‘Non-Adeno/Squamous Carcinoma’.

    Parameters
    ----------
    data : pandas.Series
        Input list of histologies (free-text/string descriptions, not histology
        codes).

    Returns
    -------
    list[str]
        List of broad histology groups.

    Notes
    -----
    Mapping is performed using pre-defined patterns, therefore, patterns may
    need to be updated from time to time to accommodate novel histology patterns.
    """

    import pandas as pd
    import numpy as np

    # Make sure input is pandas series
    data = pd.Series(data)

    # Define histology grouping dictionary
    histology_groupings = {
        # -----------------------------------------
        # Neuroendocrine Neoplasms (NET / NEC)
        # -----------------------------------------
        "Neuroendocrine Neoplasm": [
            r"\bneuroendocrine\b",
            r"\bcarcinoid\b",
            r"(?<!non-)small cell\b",
            r"\blarge cell.*neuroendocrine\b",
            r"\bmerkel\b",
            r"\bpheochromocytoma\b",
            r"\bparaganglioma\b",
            # Sex-cord stromal tumors
            r"\bsex cord",
            r"\bgranulosa\b",
        ],
        # ---------------------------
        # Melanoma or Melanocytic
        # ---------------------------
        "Melanoma or Melanocytic": [
            r"\bmelanoma\b",
            r"\bmelanocytic\b",
            r"\buveal\b",
        ],
        # ----------------------------------------------
        # CNS tumors (glioma spectrum + meningioma)
        # ----------------------------------------------
        "CNS Tumor": [
            r"\bglioblast",
            r"\bgliosarcoma\b",
            r"\bglioma\b",
            r"\bastrocyt",
            r"\boligodendrogli",
            r"\bependym",
            r"\bneurocytoma\b",
            r"\bcns.*embryonal\b",
            r"\bcentral nervous system.*embryonal\b",
            r"\bmeningioma\b",
            r"\bbrain\b",
            r"\bneuroepithelial\b",
            r"\bneuroblastic\b",
        ],
        # ---------------------
        # Germ Cell Tumors
        # ---------------------
        "Germ Cell Tumor": [
            r"\bgerm cell\b",
            r"\bseminoma\b",
            r"\byolk sac\b",
            r"\bteratoma\b",
            r"\bmixed germ\b",
        ],
        # --------------------------
        # Sarcoma (mesenchymal)
        # --------------------------
        "Sarcoma": [
            r"\bsarcoma\b",
            r"\bewing\b",
            r"\bosteosarcoma\b",
            r"\bchondrosarcoma\b",
            r"\bliposarcoma\b",
            r"\bleiomyosarcoma\b",
            r"\brhabdomyosarcoma\b",
            r"\bfibrosarcoma\b",
            r"\bmyxofibro",
            r"\bsolitary fibrous\b",
            r"\bdesmoid\b",
            r"\binflammatory myofibroblastic\b",
            r"\bdesmoplastic small round cell\b",
            r"\balveolar soft part\b",
            r"\bperivascular epithelioid\b",
            r"\bpecoma\b",
            r"\bangiosarcoma\b",
            r"\bhemangioendothelioma\b",
            r"\bkaposi\b",
            r"\bgastrointestinal stromal\b",
            r"\bgist\b",
            r"\bmesothelioma\b",
            r"\bmesothelial\b",
            r"\bphyllodes\b",
            r"\bschwannoma\b",
            r"\bperipheral nerve\b",
            r"\bspindle cell\b",
            r"\bhistiocyt",
            r"\blangerhans\b",
            r"\bdendritic cell\b",
            r"\bgiant cell\b",
            r"\bfibrohistiocytic\b",
            r"\bvascular neoplasm\b",
            r"\btenosynovial\b",
            r"\bchordoma\b",
            r"\bsmooth muscle neoplasm\b",
            # Soft tissue NOS -> Sarcoma
            r"\bsoft tissue tumor of uncertain differentiation\b",
        ],
        # ---------------------------------------------------------
        # Adenocarcinoma (malignant glandular epithelium only)
        # ---------------------------------------------------------
        "Adenocarcinoma": [
            r"\badenocarcinoma\b",
            r"\bductal carcinoma\b",
            r"\blobular carcinoma\b",
            # Morphology tokens — protected from ovarian tumor & RCC leakage
            r"\bendometrioid\b(?!.*\btumor\b)",
            r"\bmucinous\b(?!.*\btumor\b)",
            r"\bclear cell\b(?!.*\btumor\b)(?!.*\brenal\b)",
            r"\bserous adenocarcinoma\b",
            # Papillary — exclude thyroid & renal
            r"\bpapillary carcinoma\b(?!.*\bthyroid\b)(?!.*\brenal\b)(?!.*\btumor\b)",
            r"\bcystadenocarcinoma\b",
            r"\bhidradenocarcinoma\b",
            r"\bsalivary gland[- ]type carcinoma\b",
        ],
        # ----------------------------
        # Squamous Cell Carcinoma
        # ----------------------------
        "Squamous Cell Carcinoma": [
            r"\bsquamous\b",
            r"\bbasaloid carcinoma\b",
        ],
        # --------------------------------------------------------
        # Non-Adeno or Squamous Carcinoma (GENIE-like catch-all)
        # --------------------------------------------------------
        "Non-Adeno or Squamous Carcinoma": [
            r"\burothelial\b",
            r"\brenal\b",
            r"\brenal cell carcinoma\b",
            r"\bclear cell renal\b",
            r"\bpapillary renal\b",
            r"\bsarcomatoid renal\b",
            # Hepatobiliary
            r"\bhepatocellular carcinoma\b",
            r"\bcholangiocarcinoma\b",
            # Thyroid (all forms)
            r"\bthyroid\b",
            # Thymic epithelial
            r"\bthymic carcinoma\b",
            r"\bthymoma\b",
            # Salivary epithelial (non-adeno)
            r"\badenoid cystic\b",
            r"\bmucoepidermoid\b",
            r"\bacinic cell\b",
            r"\bepithelial[- ]myoepithelial\b",
            r"\bsalivary duct\b",
            # Ovarian borderline / non-malignant epithelial
            r"\bovarian .* tumor\b",
            r"\bserous tumor\b",
            r"\bepithelial tumor\b",
            r"\bborderline\b",
            # Mixed epithelial–mesenchymal
            r"\badenosquamous\b",
            r"\bcarcinosarcoma\b",
            r"\badenosarcoma\b",
            r"\bmixed mesodermal\b",
            r"\bmullerian\b",
            r"\bsarcomatoid carcinoma\b",
            r"\bundifferentiated carcinoma\b",
            # Generic fallbacks (must remain last)
            r"\bcarcinoma\b",
            r"\bneoplasm\b",
            r"\btumor\b",
        ],
    }

    # Perform mapping
    map_results = np.select(
        condlist=[
            data.str.lower()
            .str.strip()
            .str.contains(
                "|".join(histology_groupings[group]),
                regex=True,
                na=False,
            )
            for group in list(histology_groupings.keys())
        ],
        choicelist=list(histology_groupings.keys()),
        default=pd.NA,
    )

    return map_results
