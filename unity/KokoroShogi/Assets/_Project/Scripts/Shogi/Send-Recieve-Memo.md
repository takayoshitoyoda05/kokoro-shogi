
# 送受信関連のメモ

## サーバーを立ち上げるとき

## データの形式

### 1.対局開始：game_control

Unity→AI
{ "schema": "1.0", "type": "game_control", "command": "start" }

### 2.pythonからUnityへ盤の情報を送信：state_update（駒の情報）, legal_moves(合法手を表示)

//P:歩orと金, L:香or成香, N:桂or成桂 S:銀or成銀, G:金, B:角or馬, R:飛or龍, K,玉or王
//sfenについて: 数字は直前の文字が示す駒の数を表している
//

{
    "schema": "1.0",
    "type": "state_update",
    "ply": 42,　                                        //何手目か
    "sfen": "ln2g1snb/r1k2g2l/... b - 43", 　　　　　　　//行ごとの駒の種類と数
    "last_move": {
    "from": "32", "to": "23", //ここを取得する
    "piece_id": "G49_gen0_0018",
    "capture": false, "promote": false, "drop": false //あとここ
},
    "eval": -0.38,
    "pieces": [
        {
            "piece_id": "B22_gen0_0006", 　　　　　　　　　　　　　　　//piece_idの割り当てがわからん
            "species": "KA", "owner": 1, "square": "11",
            "mood": { "fear": 0.175, "aggression": 0.15, "valence": 0.34 },
            "desire": { "survive": 0.387, "attack": 0.15, "promote": 0.0,
            "defend": 0.0, "advance": 0.0, "redeploy": 0.0 },
            "alpha": 0.051,
            "relations": [ { "to": "P33_gen0_0012", "r": 0.6 } ]
        }
    ]
}

//エフェクト表示へのルールづけ
駒ごとにfearとagressionを比較し、値の大きい方をエフェクトの出力としてゲームに反映させる。値の小さい方に関連するエフェクト（オブジェクト）は非表示にする。

fear値が大きい場合：
① 全ての下のDebuffオブジェクトを表示し、そのオブジェクト及び配下のparticlesystemをPlayする。
② 全てのparticlesystemのStartColorのα値を、fear値を参照して反映させる。
受信値が小さいと見えづらいため、fearを0～1に制限してから平方根（ルート）で補正する。
UnityのColor.a（0～1）：α = sqrt(Clamp01(fear値))
Inspectorの0～255表記：α = 255 * sqrt(Clamp01(fear値))
C#：float fearAlpha = Mathf.Sqrt(Mathf.Clamp01(fear));
例：fear = 0.1 → α ≒ 0.316（約81/255）、fear = 0.25 → α = 0.5（約128/255）。
fearとaggressionの大小比較は補正前の値で行う。同値・未受信では両方を非表示にする。
③ Debuffの表示サイズを、Prefabの初期LocalScaleの1.5倍にする。
Debuff.localScale = 初期LocalScale * 1.5
受信ごとに現在のScaleへ掛け算せず、初期Scaleを基準にする。透明度のルート補正とは別にサイズを拡大する。

agression値が大きい場合：
①vfx_Fireオブジェクトを表示
②agression値を参照し、vfx_FireオブジェクトのTransformのScaleを13～77で変化させる
scale = 13 + (77 - 13) * sqrt(Clamp01(aggression値))
炎のScaleにもルート補正を適用し、小さいaggression値でも大きさが出るようにする。範囲は13～77。
例：aggression = 0.15 → scale ≒ 37.8（補正前は22.6）、aggression = 0.25 → scale = 45。

チームごとの各駒のvalance値の総合値を比較し、ゲーム画面上部の形勢バーに反映する。
先手のvalanceの総合値 : 後手のvalanceの総合値 = Slider_Fill_1PValueオブジェクトのValue : Slider_Fill_2PValueオブジェクトのValue
受信JSONのフィールド名は mood.valence。owner = 0 が先手、owner = 1 が後手。
持ち駒も含む全piecesを現在のownerで集計し、局面の反映と同時にバーを更新する。
先手合計 A = max(0, Σ先手のvalence)、後手合計 B = max(0, Σ後手のvalence)。
valenceは負にもなるため、各駒ではなく陣営の合計後に0以上へ制限する。
A + B > 0 の場合：Slider_Fill_1PValue.value = A / (A + B)、Slider_Fill_2PValue.value = B / (A + B)。
両方0以下、未受信、再戦開始時はそれぞれ0.5。例：A = 3、B = 1 → 0.75 : 0.25。
エフェクト用のルート補正は使用しない。mood欠落・非数値のvalenceは集計から除外する。
局面更新時のバー移動はDOTweenで0.4秒、Ease.OutCubicを使用する（InspectorのValence Bar Animation Secondsで調整可能）。
1本のTweenで先手の比率を動かし、後手は常に1－先手として更新する。途中で次の更新が来たら、現在の表示位置から新しい比率へ移動する。
時間停止中もUIアニメーションは進める。起動・再戦時の0.5へのリセットは即時反映し、無効化・破棄時にはTweenを停止する。

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

//持ち駒を打つ場合はfrom: "00"とdrop_speciesを送る

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
