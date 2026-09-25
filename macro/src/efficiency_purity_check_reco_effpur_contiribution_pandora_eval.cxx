// Copied from efficiency_purity_check_reco_effpur_contiribution.cxx and
// adapted for pandora-comparison-root-1. The original file is unchanged.
// Event spectra use full PFO energy / common model inputs, H5 event selection,
// and the unmodified LCPandoraAnalysis RMS90 implementation. The old hit-derived
// energy reconstruction and hard-coded file lists must not run on this schema.
//
// root -l -b -q 'macro/src/efficiency_purity_check_reco_effpur_contiribution_pandora_eval.cxx("input_eval.root","performance.root")'
#include "pandora_eval_analysis.h"

void efficiency_purity_check_reco_effpur_contiribution_pandora_eval(
    const char *input, const char *output="pandora_eval_performance.root")
{
    pandora_eval::Run(input, output);
}
