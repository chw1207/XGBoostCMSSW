// #ifndef XGBREADER_H_
// #define XGBREADER_H_

#include <iostream>
#include <string>
#include <vector>
#include <xgboost/c_api.h>


class XGBReader{
public:
    XGBReader(std::string _fName);
    ~XGBReader();

    std::vector<float> Compute(const std::vector<float>& features);

private:
    void safe_xgboost(uint64_t call);

    BoosterHandle booster;

    // new version v > 1
    // c_json_config 
    // const char* config = "{\"training\": false, \"type\": 0, \"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": true}";
};
// #endif // XGBREADER_H_
