文件说明
app.py（后端接口）
功能：基于 Flask 开发的后端接口服务，主要提供以下能力：
处理跨域请求，确保前端可正常调用；
提供 /api/long-term-memory 接口，调用 CreateLongTermMemories.py 脚本，实现长期记忆存储；
提供 /api/rag-generate 核心接口，对接 RAG 系统的三个应用（输入优化、菜谱检索、烹饪步骤生成），返回结构化结果；
兼容原有 /api/search-qa 接口，适配前端历史调用逻辑。
依赖：需安装 flask、flask-cors、dashscope 等库，可通过 pip install flask flask-cors dashscope 安装。

App.vue（前端界面）
功能：基于 Vue 开发的前端交互界面，主要包含：
用户输入区域：支持输入问题、指定食材、禁止食材、不可用厨具；
结果展示区域：以打字机效果展示 RAG 系统返回的 “输入优化、菜谱列表、烹饪步骤”；
交互按钮：“生成菜谱” 触发 RAG 调用，“继续对话” 支持多轮交互（剩余轮次递减），“重置” 清空所有状态；
样式与响应式：适配不同设备，确保界面美观易用。

使用方法
启动后端
确保安装好所有依赖（flask、flask-cors、dashscope）；
直接运行 app.py：
python app.py
后端将在 http://0.0.0.0:5001 启动，自动处理跨域请求。
启动前端
确保 Vue 开发环境已配置（如安装 @vue/cli）；
进入前端项目目录，启动开发服务器：
bash
npm run serve
访问前端地址（默认 http://localhost:8080），即可开始使用。