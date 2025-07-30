def compliance_images_cmd(loadgen_dataset_size, compliance_images_path):
    if loadgen_dataset_size == 5000:
        cmd = "--compliance-images-path " + compliance_images_path
    else:
        cmd = ""
    return cmd
