"""駒の個体データを永続化するSQLiteストア。

実装予定:
- ストア: θ_ind (性格個体値) / career (キャリア実績) / lineage (血統) の
  保存・読み出し・世代をまたいだ引き継ぎ
- features.yaml の individual フラグと連動した読み込み

参照: DESIGN.md §7 永続化とキャリア・血統
実装: 週7 / 担当: A
"""
