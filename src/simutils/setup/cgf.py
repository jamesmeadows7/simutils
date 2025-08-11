FORCE_FIELDS = {
    "opls":["chelpg_chelpg", "chelpg_cm1a", "ligpargen_chelpg", "ligpargen_cm1a", "opls2020_basic", "opls2020_chelpg", "opls2020_cm1a", "opls2020_cndo", "opls_basic", "opls_chelpg", "opls_cm1a", "opls_cndo"],
    "gaff":["gaff2_abcg2", "gaff2_cndo", "gaff2_dnp", "gaff2_resp", "gaff_bcc", "gaff_cndo", "gaff_dnp", "gaff_resp"],
}

POLYMORPHS = ["alpha", "beta", "gamma"]

DEFAULTS = {
    "opls":"    1         3          yes         0.5         0.5",
    "gaff":"    1         2          yes         0.5        0.8333"
}

BETA_ZX = {"alpha":"1e-5", "beta":"1e-5", "gamma":"0.0"}

Z = {"alpha": 4, "beta": 2, "gamma": 3}

WATER_MODELS = ["tip3p", "tip4p", "tip4p2005"]

TEMPS = [298.15, 318.15] # K