
# masakiメモ

## サーバーを立ち上げるとき

## データの形式

### 1.対局開始：game_control

Unity→AI
{ "schema": "1.0", "type": "game_control", "command": "start" }

### 2.pythonからUnityへ盤の情報を送信：state_update（駒の情報）, legal_moves(合法手を表示)

{
    "schema": "1.0",
    "type": "state_update",
    "ply": 42,　                                        //何手目か
    "sfen": "ln2g1snb/r1k2g2l/... b - 43", 　　　　　　　//行ごとの駒の種類と数
    "last_move": {
    "from": "32", "to": "23",
    "piece_id": "G49_gen0_0018",
    "capture": false, "promote": false, "drop": false
},
    "eval": -0.38,
    "pieces": [
        {
            "piece_id": "B22_gen0_0006",
            "species": "KA", "owner": 1, "square": "11",
            "mood": { "fear": 0.175, "aggression": 0.15, "valence": 0.34 },
            "desire": { "survive": 0.387, "attack": 0.15, "promote": 0.0,
            "defend": 0.0, "advance": 0.0, "redeploy": 0.0 },
            "alpha": 0.051,
            "relations": [ { "to": "P33_gen0_0012", "r": 0.6 } ]
        }
    ]
}

{
    "schema": "1.0",
    "type": "legal_moves",
    "moves": [
        { "from": "77", "to": "76", "promote": false },
        { "from": "88", "to": "22", "promote": true },
        { "from": "00", "to": "55", "drop_species": "FU" }
    ]
}

### 3.プレイヤーが選んだ手をpythonに送信：move_request

Unity → AI
{
    "schema": "1.0",
    "type": "move_request",
    "move": { "from": "77", "to": "76", "promote": false }
}

### 4.AIが思考した結果の盤情報がUnityへ送られてくる（プレイヤーの番 → AIの番の計2回送られてくる）一つのファイルに2コ?：state_update

{
    "schema": "1.0",
    "type": "state_update",
    "ply": 42,　                                        //これは何手目か
    "sfen": "ln2g1snb/r1k2g2l/... b - 43", 　　　　　　　//行ごとの駒の種類と数。bは先手、wは後手の情報。一番後ろのブロックは持ち駒の情報。例）2Pb → b(先手)がPを2コ持っている。
    "last_move": {
    "from": "32", "to": "23",　　　　　　　　　　　　　　//ここ（駒の移動情報）と
    "piece_id": "G49_gen0_0018",
    "capture": false, "promote": false, "drop": false  //ここ（駒を獲ったか、成ったか、dropってなんや…）を使えばよさそう
},
    "eval": -0.38,                                     //エフェクトの強弱に影響を与える値（追々やる）
    "pieces": [
        {
            "piece_id": "B22_gen0_0006",
            "species": "KA", "owner": 1, "square": "11",
            "mood": { "fear": 0.175, "aggression": 0.15, "valence": 0.34 },
            "desire": { "survive": 0.387, "attack": 0.15, "promote": 0.0,
            "defend": 0.0, "advance": 0.0, "redeploy": 0.0 },
            "alpha": 0.051,
            "relations": [ { "to": "P33_gen0_0012", "r": 0.6 } ]
        }
    ]
}

### 終局時に戦績を表示（AI → Unity）：career

{
    "schema": "1.0",
    "type": "career",
    "pieces": [
        { "piece_id": "R28_gen0_0011", "species": "HI",
            "games": 120, "survival_rate": 0.78,           //games(何手で終わったか),...
            "promotions": 55, "mvp_count": 21 }
    ],
    "last_game_mvp": { "piece_id": "R28_gen0_0011", "contribution": 2.3 }
}

### あああ

あ
