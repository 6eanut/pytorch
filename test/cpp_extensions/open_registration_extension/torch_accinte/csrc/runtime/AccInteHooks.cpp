#include "AccInteHooks.h"

// LITERALINCLUDE START: ACCINTE HOOK REGISTER
namespace c10::accinte {

static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new AccInteHooksInterface());

  return true;
}();

} // namespace c10::accinte
// LITERALINCLUDE END: ACCINTE HOOK REGISTER