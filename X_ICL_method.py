import json
import openai
import os
import pandas as pd

qwen_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "sk-2c1c06ba06b341d2b10aee6cd9540924"

def get_response(premise, hypothesis, label) -> str:
    openai.api_key = qwen_url
    model_name = "qwen-turbo"

    # Step 1A: 使用少样本提示生成NLE
    # 提供几个已经标记好的NLI示例，作为演示
    examples = [
        {
            "premise": "A boy peers out of an open window.",
            "hypothesis": "The boy looks out the window.",
            "label": "entailment",
            "nle": "The boy peers out of a window, so the boy looks out the window."
        },
        {
            "premise": "A man in a jean jacket is sitting outside painting.",
            "hypothesis": "There is a man outside.",
            "label": "entailment",
            "nle": "The premise states that a man in a jean jacket is sitting outside painting, which implies that there is a man outside."
        }
    ]

    # 构建提示消息
    messages = [
        {'role': 'system', 'content': '''
            你是一名专业的自然语言推理专家，
            你的任务是根据给定的前提、假设和标签生成自然语言解释(NLE)。
            请参考以下示例来生成NLE。
        '''},
        {'role': 'user', 'content': f'''
            根据以下前提、假设和标签生成NLE:
            Premise: {premise}
            Hypothesis: {hypothesis}
            Label: {label}
        '''}
    ]

    # 添加示例演示
    for example in examples:
        messages.append({
            'role': 'assistant',
            'content': f'''
                Premise: {example['premise']}
                Hypothesis: {example['hypothesis']}
                Label: {example['label']}
                NLE: {example['nle']}
            '''
        })

    # 请求OpenAI API生成NLE
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=messages
    )

    # 提取生成的NLE
    generated_nle = completion.choices[0].message.content

    # Step 2: 使用生成的NLE进行上下文学习
    # 构建上下文学习的提示消息
    context_learning_messages = [
        {'role': 'system', 'content': '''
            你是一名专业的自然语言推理模型，
            你的任务是根据给定的前提、假设和自然语言解释(NLE)来预测标签。
            请参考以下示例来预测标签。
        '''},
        {'role': 'user', 'content': f'''
            根据以下前提、假设和NLE预测标签:
            Premise: {premise}
            Hypothesis: {hypothesis}
            NLE: {generated_nle}
        '''}
    ]

    # 添加示例演示
    for example in examples:
        context_learning_messages.append({
            'role': 'assistant',
            'content': f'''
                Premise: {example['premise']}
                Hypothesis: {example['hypothesis']}
                NLE: {example['nle']}
                Label: {example['label']}
            '''
        })

    # 请求OpenAI API进行上下文学习
    context_learning_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=context_learning_messages
    )

    # 提取预测的标签
    predicted_label = context_learning_completion.choices[0].message.content
    
    # 打印生成的NLE和预测的标签
    print("Generated NLE:", generated_nle)
    print("Predicted Label:", predicted_label)

    return generated_nle, predicted_label

if __name__ == '__main__':
    # 为./data/twitter.csv下所有新闻内容（post_text）生成nle，存储到./data/xicl_nle.csv内
    df = pd.read_csv('./data/twitter.csv')
    df['nle'] = ''
    df['predicted_label'] = ''
    for i in range(len(df)):
        premise = df.loc[i, 'post_text']
        hypothesis = "The post is about a news."
        label = "entailment"
        print("Generating NLE and predicted label for premise:", premise)
        generated_nle, predicted_label = get_response(premise, hypothesis, label)
        df.loc[i, 'nle'] = generated_nle
        df.loc[i, 'predicted_label'] = predicted_label
    df.to_csv('./data/xicl_nle.csv', index=False)
    print("NLE and predicted label generated successfully!")
