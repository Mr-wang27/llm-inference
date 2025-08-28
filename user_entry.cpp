#include <stdio.h>
#include "src/utils/model_utils.h"
#include "src/models/basemodel.h"
#include <string>
#include <iostream>

struct ConvertedModel{
    std::string model_path = "/home/llamaweight/";  // 模型文件路径
    std::string tokenizer_path = "/home/llama2-7b-tokenizer.bin";   // tokenizer文件路径
};


int main(int argc, char** argv)
{
    int round = 0;              // 对话轮次
    std::string history = "";   // 历史信息
    ConvertedModel model;       // 模型相关文件地址信息

    // 加载模型模型以及tokenizer（创建出模型的计算图以及加载模型权重和tokenizer权重）
    // auto llm_model = llm::CreateRealLLMModel<float>(model.model_path, model.tokenizer_path);
    auto llm_model = llm::CreateDummyLLMModel<float>(model.tokenizer_path);
    std::string model_name = llm_model->model_name; 

    // exist when generate and token ot reach max seq
    while(true){
        printf("please input the queston: ");
        std::string input;
        std::getline(std::cin, input);
        if(input == "exit"){// 停止对话
            break;    
        }

        // index表示生成的第几个toekn,从0开始
        std::string resString = llm_model->Response(llm_model->MakeInput(history, round, input), [model_name](int index, const char* content){
            if(index == 0){
                printf(":%s", content);
                fflush(stdout); // 刷新输出缓冲区
            }
            if(index > 0){
                printf("%s", content);
                fflush(stdout);
            }
            if(index == -1) {// 表示没有生成数据
                printf("\n");
            }
        });

        // 完成生成后，更新history
        history = llm_model->MakeHistory(history, round, input, resString);
        round++;
    }
    return 0;
}