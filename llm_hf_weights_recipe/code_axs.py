def get_model_info(model_family, variant, model_family_information=None, __entry__=None):
    model_info = model_family_information[model_family]
    
    assert "name_format" in model_info, f"Model family {model_family} has no \"name_format\" field"
    assert "model_publisher" in model_info, f"Model family {model_family} has no \"model_publisher\" field"
    assert "allowed_variants" in model_info, f"Model family {model_family} has no \"allowed_variants\" field"
    
    assert variant in model_info["allowed_variants"], f"Cannot find {model_family}-{variant}"
    
    return {
        "model_name": __entry__.substitute(model_info["name_format"]),
        "model_publisher": model_info["model_publisher"]
    }
