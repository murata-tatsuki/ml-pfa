/**
 * Native executable entry point for ../src/efficiency_purity_check_reco_effpur_contiribution.cxx
 * Lives under macro/mains/ (see macro/README_BUILD.md).
 *
 * バッチモード固定: ウィンドウは出さない（root hoge.cxx のインタプリタ実行とは別）。
 * 図の保存は saving_canvas や SaveAs がバッチでも動く。
 */
#include "TApplication.h"
#include "TError.h"
#include "TROOT.h"

void efficiency_purity_check_reco_effpur_contiribution();

int main(int argc, char** argv)
{
    // 重複ロード時の "TClassTable::Add ... already in TClassTable" 等を非表示にする。
    // 根本対処は README_BUILD.md「ROOT 実行環境」のとおり LD_LIBRARY_PATH を一本化すること。
    gErrorIgnoreLevel = kError;

    gROOT->SetBatch(kTRUE);
    TApplication app("effpur_contribution", &argc, argv);
    efficiency_purity_check_reco_effpur_contiribution();
    // app.Run() は GUI 用イベントループ — バッチでは不要
    return 0;
}
