#include <algorithm>
// #include "TString.h"
#include "XGBoostCMSSW/XGBoostInterface/interface/XGBReader.h"



XGBReader::XGBReader(std::string _fName){
    safe_xgboost(XGBoosterCreate(NULL, 0, &booster));

    // set the nthread to 1 to have faster single event inference
    safe_xgboost(XGBoosterSetParam(booster, "nthread", "1"));
    safe_xgboost(XGBoosterLoadModel(booster, _fName.c_str()));
}


XGBReader::~XGBReader(){
    safe_xgboost(XGBoosterFree(booster));
}


std::vector<float> XGBReader::Compute(const std::vector<float>& features){
    // old version v0.8
    DMatrixHandle dpred;
    bst_ulong out_shape = 0;  
    float const *out_results;  
    safe_xgboost(XGDMatrixCreateFromMat((float*)features.data(), 1, features.size(), -99999., &dpred));
    safe_xgboost(XGBoosterPredict(booster, dpred, 0, 0, &out_shape, &out_results));
    std::vector<float> score(out_results, out_results + out_shape);

    // new version v > 1
    /*
        DMatrixHandle dpred;
        uint64_t const *out_shape;
        float const *out_results;
        uint64_t out_dim;
        safe_xgboost(XGDMatrixCreateFromMat((float*)features.data(), 1, features.size(), -99999., &dpred));
        safe_xgboost(XGBoosterPredictFromDMatrix(booster, dpred, config, &out_shape, &out_dim, &out_results));
        printf("dim: %lu, shape[0]: %lu, shape[1]: %lu\n", out_dim, out_shape[0], out_shape[1]);
        std::vector<float> score(out_results, out_results + out_shape[1]);
    */

    safe_xgboost(XGDMatrixFree(dpred));

    return score;
}


void XGBReader::safe_xgboost(uint64_t call){
    if (call != 0)
        throw std::runtime_error(XGBGetLastError());
}
