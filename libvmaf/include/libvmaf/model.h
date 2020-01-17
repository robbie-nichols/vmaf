#ifndef __VMAF_MODEL_H__
#define __VMAF_MODEL_H__

#include <libvmaf/model.config.h>

typedef struct VmafModel VmafModel;

int vmaf_model_load_from_path(VmafModel **model, VmafModelConfig *config);
void vmaf_model_destroy(VmafModel *model);

#endif /* __VMAF_MODEL_H__ */
