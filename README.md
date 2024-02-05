# udemy-langchain
udemyの学習コンテンツに公式チュートリアルの実装も加えています

##　開発環境
- venv環境で開発することを想定しています
- requirements.txtのファイルをインストールして使っています。ただし、追々パッケージ管理に移行する予定です

##　操作方法
- venv環境をアクティブにする
- srcファイル内にpyファイルを作成し、python pyファイル名で実行することで動作します

##　実装内容（srcファイル内のpyファイル）
- retrievalchain.pyで、RAGを利用した検索を行っています
- slack_app.pyで、Udemynの教材を参考に、slackのbotとして動作するように実装しています
- rag.pyは「LangChainとRAG: Embeddingで外部データを利用する」を参考に実装しています
- slack_app.py
- code_understanding.pyでは、公式のユースケースを参考に、Pythonのコードの理解を行っています
  repo_path +の後ろにリポジトリを指定することで、そのリポジトリのコードを理解することができます
- code_understanding_typescript.pyは、Next.jsのコードを理解するためにTSをターゲットに実装していますが、2024/2/5現在動きません

##　メモ


###　RAG参考
<!-- markdownでURLリンクを作成 -->
[LangChainとRAG: Embeddingで外部データを利用する](https://developers.cyberagent.co.jp/blog/archives/44973/)
[LangChainテンプレート集](https://templates.langchain.com/)
[LangChainテンプレート集解説](https://hironsan.hatenablog.com/entry/research-assistant-with-langchain)