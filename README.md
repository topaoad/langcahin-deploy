# udemy-langchain

udemy の学習コンテンツに公式チュートリアルの実装も加えています

##　開発環境

- venv 環境で開発することを想定しています
- requirements.txt のファイルをインストールして使っています。ただし、追々パッケージ管理に移行する予定です

##　操作方法

- venv 環境をアクティブにする
- src ファイル内に py ファイルを作成し、python py ファイル名で実行することで動作します

##　実装内容（src ファイル内の py ファイル）

- retrievalchain.py で、RAG を利用した検索を行っています
- slack_app.py で、Udemyn の教材を参考に、slack の bot として動作するように実装しています
- quickstart.py は、公式のチュートリアルを参考に、RAG を利用した検索を行っています
  　 Agent も使われていて一通りの実装が備わっています。
- rag.py は「LangChain と RAG: Embedding で外部データを利用する」を参考に実装しています
- rag_usecase.py ha は、公式のユースケースを参考に、RAG を利用した検索を行っています
- slack_app.py
- code_understanding.py では、公式のユースケースを参考に、Python のコードの理解を行っています
  repo_path +の後ろにリポジトリを指定することで、そのリポジトリのコードを理解することができます
- code_understanding_typescript.py は、Next.js のコードを理解するために TS をターゲットに実装していますが、2024/2/5 現在動きません
- web_scraping.py はウェブスクレイピングを行うためのコードですが、2024/2/5 現在動きません

##　メモ
ユースケースで使われているプロンプトやメモリの設定は、RunnableWithMessageHistory など LCEL に置き換わっていくものと思われる。

##　課題(2024/2/6)
- web_scraping.pyの検証が途中
- Agentをもう少し理解したい
- 全体的にそれぞれのパーツをどう組み合わせて活かすかをより考えたい
- 

###　 RAG 参考

<!-- markdownでURLリンクを作成 -->

[LangChain と RAG: Embedding で外部データを利用する](https://developers.cyberagent.co.jp/blog/archives/44973/)
[LangChain テンプレート集](https://templates.langchain.com/)
[LangChain テンプレート集解説](https://hironsan.hatenablog.com/entry/research-assistant-with-langchain)
[Agentを使ったコード解読参考※まだ未実装](https://note.com/kioju/n/n3d3401cd499f)
