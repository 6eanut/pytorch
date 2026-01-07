#pragma once

#ifdef _WIN32
#define ACCINTE_EXPORT __declspec(dllexport)
#else
#define ACCINTE_EXPORT __attribute__((visibility("default")))
#endif
