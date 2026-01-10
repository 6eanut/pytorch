#include <Python.h>

#ifdef _WIN32
#define ACCINTE_EXPORT __declspec(dllexport)
#else
#define ACCINTE_EXPORT __attribute__((visibility("default")))
#endif

extern ACCINTE_EXPORT PyObject* initAccInteModule(void);

#ifdef __cplusplus
extern "C"
#endif

    ACCINTE_EXPORT PyObject*
    PyInit__C(void);

PyMODINIT_FUNC PyInit__C(void) {
  return initAccInteModule();
}
