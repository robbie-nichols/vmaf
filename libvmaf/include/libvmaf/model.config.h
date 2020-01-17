#ifndef __VMAF_MODEL_CONFIG_H__
#define __VMAF_MODEL_CONFIG_H__

enum VmafModelFlags {
    VMAF_MODEL_FLAG_DISABLE_CLIP = (1 << 0),
    VMAF_MODEL_FLAG_ENABLE_TRANSFORM = (1 << 1),
    VMAF_MODEL_FLAG_ENABLE_CONFIDENCE_INTERVAL = (1 << 2),
    VMAF_MODEL_FLAG_DEFAULT = VMAF_MODEL_FLAG_DISABLE_CLIP,
};

typedef struct VmafModelConfig {
    enum VmafModelFlags flags;
    char *name;
    char *path;
} VmafModelConfig;

void vmaf_model_config_destroy(VmafModelConfig *config);

#endif /* __VMAF_MODEL_CONFIG_H__ */
