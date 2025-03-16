from retrieval import VectorRetriever
from openai import OpenAI
from config.settings import Config
import gradio as gr
from init_data import initialize_agents

def get_openai_response(query, sys_prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

def process_result(results):
     # 格式化输出结果
    output = []
    best_match = None
    for i, res in enumerate(results, 1):
        if i == 1:
            best_match = res
        result_str = (
            f"结果 {i}:\n"
            f"ID: {res['id']}\n"
            f"相似度: {res['score']:.4f}\n"
            f"内容: {res['document'][:80]}...\n"
            f"{'='*50}"
        )
        output.append(result_str)
    agent_search_results = "\n\n".join(output)
    return agent_search_results, best_match

def process_query_first(query):
    retriever = VectorRetriever()
    results = retriever.enhanced_search(query)["search_results"]
    agent_search_results, best_match = process_result(results)
    ai_response = get_openai_response(query, best_match['document'])
    return agent_search_results, ai_response

def process_query_second(agent_search_results, query, new_category=None):
    # if feedback == "满意":
    #     return agent_search_results, ai_response
    
    # elif feedback == "不满意":

    if new_category:
        # 初始化新的agent类型
        new_agents = initialize_agents([new_category], client)
        # 重新进行检索
        return agent_search_results, get_openai_response(query, new_agents[0]["system_prompt"])
    else:
        results = get_openai_response(query, "你是一个AI助手")
        return agent_search_results, results

def main():
    with gr.Blocks() as demo:
        # 主要输入组件
        with gr.Row():
            query_input = gr.Textbox(label="请输入您的问题", lines=3)
            search_btn = gr.Button("查询")

        # 显示检索结果和回答
        with gr.Row():
            agent_results = gr.Textbox(label="检索到的Agent", lines=8)
            ai_response = gr.Textbox(label="AI回答", lines=8)

        # 反馈按钮
        with gr.Row():
            satisfied_btn = gr.Button("满意")
            unsatisfied_btn = gr.Button("不满意")

        # 新建agent类型的输入框（初始隐藏）
        with gr.Row(visible=False) as new_agent_row:
            new_category_input = gr.Textbox(label="请输入新的Agent类型")
            submit_new_category = gr.Button("提交")

        # 存储当前最佳匹配结果的状态
        state = gr.State(value=None)
        
        def show_new_category_input():
            return {new_agent_row: gr.update(visible=True)}

        # 处理首次查询
        def handle_first_query(query):
            agent_results_text, response = process_query_first(query)
            state.value = response
            return {
                agent_results: agent_results_text,
                ai_response: response,
            }

        # 处理满意按钮
        def handle_satisfied(agent_results_text, query):
            if state.value:
                _, response = process_query_second(
                    agent_results_text, 
                    query, 
                    state.value, 
                )
                return {ai_response: response}
            return {ai_response: "请先进行查询"}

        # 处理新类型提交
        def handle_new_category(agent_results_text, query, new_category):
            _, response = process_query_second(
                agent_results_text,
                query,
                new_category=new_category
            )
            return {
                ai_response: response,
                new_agent_row: gr.update(visible=False)
            }

        # 设置点击事件
        search_btn.click(
            handle_first_query,
            inputs=[query_input],
            outputs=[agent_results, ai_response]
        )

        satisfied_btn.click(
            handle_satisfied,
            inputs=[agent_results, query_input],
            outputs=[ai_response]
        )

        unsatisfied_btn.click(
            show_new_category_input,
            inputs=None,
            outputs=[new_agent_row]
        )

        submit_new_category.click(
            handle_new_category,
            inputs=[agent_results, query_input, new_category_input],
            outputs=[ai_response, new_agent_row]
        )

        # Examples remain the same
        gr.Examples(
            examples=[
                ["金融投资相关的建议"],
                ["法律咨询问题"],
                ["医疗保健知识"]
            ],
            inputs=query_input
        )
    
    demo.launch(share=True)

if __name__ == "__main__":
    # 初始化数据（首次运行后可以注释掉）
    categories = ["finance", "law", "medical", "technology", "education", "career", "fashion", "travel", \
                  "politics", "entertainment", "mental health"]
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    # agent_info = initialize_agents(categories, client)
    main()

