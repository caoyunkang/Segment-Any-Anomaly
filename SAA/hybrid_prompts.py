from .parameters_table.general_prompts import build_general_prompts
from .parameters_table import visa_parameters
from .parameters_table import mvtec_parameters
from .parameters_table import ksdd2_parameters
from .parameters_table import mtd_parameters


manul_prompts = {
    'visa_public': visa_parameters.manual_prompts,
    'visa_challenge': visa_parameters.manual_prompts,
    'mvtec': mvtec_parameters.manual_prompts,
    'ksdd2': ksdd2_parameters.manual_prompts,
    'mtd': mtd_parameters.manual_prompts,

}

property_prompts = {
    'visa_public': visa_parameters.property_prompts,
    'visa_challenge': visa_parameters.property_prompts,
    'mvtec': mvtec_parameters.property_prompts,
    'ksdd2': ksdd2_parameters.property_prompts,
    'mtd': mtd_parameters.property_prompts,
}
