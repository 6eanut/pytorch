#include "AccInteGuard.h"

namespace c10::accinte {

// LITERALINCLUDE START: ACCINTE GUARD REGISTRATION
C10_REGISTER_GUARD_IMPL(PrivateUse1, AccInteGuardImpl);
// LITERALINCLUDE END: ACCINTE GUARD REGISTRATION
} // namespace c10::accinte
