using System;
using System.Runtime.CompilerServices;
using System.Globalization;
using System.ComponentModel;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;
using System.Collections;
using Unity.VisualScripting;
using UnityEngine.SceneManagement;
using KokoroShogi.Net;

public partial class GameSceneDirector : MonoBehaviour
{
    [SerializeField] PieceCameraAnimator pieceCameraAnimator;
    ServerBoardSynchronizer serverBoard;

    void Awake()
    {
        if (!pieceCameraAnimator)
            pieceCameraAnimator = GetComponent<PieceCameraAnimator>();
        if (!pieceCameraAnimator)
            pieceCameraAnimator = gameObject.AddComponent<PieceCameraAnimator>();
        serverBoard = GetComponent<ServerBoardSynchronizer>();
        if (!serverBoard) serverBoard = gameObject.AddComponent<ServerBoardSynchronizer>();
    }

    //UI関連
    [SerializeField] TMP_Text textTurnInfo;
    [SerializeField] TMP_Text textTurnNumber;
    [SerializeField] TMP_Text textResultInfo;
    [SerializeField] Button buttonTitle;
    [SerializeField] Button buttonRematch;
    [SerializeField] Button buttonEvolutionApply;
    [SerializeField] Button buttonEvolutionCancel;

    //ゲーム設定
    const int PlayerMax = 2;
    int boardWidth;
    int boardHeight;

    //タイルのプレハブ
    [SerializeField] GameObject prefabTile;

    //ユニットのプレハブ
    [SerializeField] List<GameObject> prefabUnits;

    //初期配置
    int[,] boardSetting =
    {
        { 4, 0, 1, 0, 0, 0, 11, 0, 14 },
        { 5, 2, 1, 0, 0, 0, 11,13, 15 },
        { 6, 0, 1, 0, 0, 0, 11, 0, 16 },
        { 7, 0, 1, 0, 0, 0, 11, 0, 17 },
        { 8, 0, 1, 0, 0, 0, 11, 0, 18 },
        { 7, 0, 1, 0, 0, 0, 11, 0, 17 },
        { 6, 0, 1, 0, 0, 0, 11, 0, 16 },
        { 5, 3, 1, 0, 0, 0, 11,12, 15 },
        { 4, 0, 1, 0, 0, 0, 11, 0, 14 },
    };

    //フィールドのデータ
    Dictionary<Vector2Int, GameObject> tiles;
    UnitController[,] units;

    //現在選択中のユニット
    UnitController selectUnit;

    //移動可能範囲
    Dictionary<GameObject, Vector2Int> movableTiles;

    //カーソルのプレハブ
    [SerializeField] GameObject prefabCursor;

    //カーソルオブジェクト
    List<GameObject> cursors;

    //プレイヤーとターン
    int nowPlayer;
    int turnCount;
    bool isCpu;

    LegalMove pendingPlayerMove;
    UnitController pendingMoveUnit;
    Vector2Int pendingMoveDestination;

    //モード
    enum Mode
    {
        None,
        Start,
        Select,
        WaitEvolution,
        Animating,
        TurnChange,
        Result
    }

    Mode nowMode, nextMode;

    //持ち駒タイルのプレハブ
    [SerializeField] GameObject prefabUnitTile;

    //持ち駒を置く場所
    List<GameObject>[] unitTiles;

    //キャプチャされたユニット
    List<UnitController> captureUnits;

    //敵陣設定
    const int EnemyLine = 3;
    List<int>[] enemyLines;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // 既存シーンでは同じCanvas内の手数表示を自動で参照する。
        if (!textTurnNumber && textTurnInfo && textTurnInfo.canvas)
        {
            foreach (TMP_Text label in textTurnInfo.canvas.GetComponentsInChildren<TMP_Text>(true))
            {
                if (label.name != "TextTurnNumber") continue;
                textTurnNumber = label;
                break;
            }
        }
        SetMoveCount(0);

        //UI関連初期設定
        buttonTitle.gameObject.SetActive(false);
        buttonRematch.gameObject.SetActive(false);
        buttonEvolutionApply.gameObject.SetActive(false);
        buttonEvolutionCancel.gameObject.SetActive(false);
        textResultInfo.text = "";

        //ボードサイズ
        boardWidth = boardSetting.GetLength(0);
        boardHeight = boardSetting.GetLength(1);

        //フィールド初期化
        tiles = new Dictionary<Vector2Int, GameObject>();
        units = new UnitController[boardWidth, boardHeight];

        //移動可能範囲
        movableTiles = new Dictionary<GameObject, Vector2Int>();
        cursors = new List<GameObject>();

        //* ここ怪しい
        //持ち駒を置く場所 //=2
        unitTiles = new List<GameObject>[PlayerMax];

        //キャプチャされたユニット
        captureUnits = new List<UnitController>();

        for (int i = 0; i < boardWidth; i++)
        {
            for (int j = 0; j < boardHeight; j++)
            {
                //タイルとユニットのポジション
                float x = i - boardWidth / 2;
                float y = j - boardHeight / 2;

                //ポジション
                Vector3 pos = new Vector3(x, 0, y);

                //タイルのインデックス
                Vector2Int tileindex = new Vector2Int(i, j);

                //タイル作成
                GameObject tile = Instantiate(prefabTile, pos, Quaternion.identity);
                tiles.Add(tileindex, tile);




                //ユニット作成
                int type = boardSetting[i, j] % 10;
                int player = boardSetting[i, j] / 10;

                if (0 == type) continue;

                //初期化
                pos.y = 0.7f;

                GameObject prefab = prefabUnits[type - 1]; //1～8のboardSettingsの値を0～7に変換して、プレハブユニットの値に割り当てる
                GameObject unit = Instantiate(prefab, pos, Quaternion.Euler(0, player * 180, 0));
                unit.AddComponent<Rigidbody>();

                UnitController unitctrl = unit.AddComponent<UnitController>();
                unitctrl.Init(player, type, tile, tileindex);

                //ユニットデータセット
                units[i, j] = unitctrl;
            }
        }

        //持ち駒を置く場所作成
        Vector3 startpos = new Vector3(5, 0.5f, -2);
        for (int i = 0; i < PlayerMax; i++)
        {
            unitTiles[i] = new List<GameObject>();
            int dir = (0 == i) ? 1 : -1;

            for (int j = 0; j < 9; j++)
            {
                Vector3 pos = startpos;
                pos.x = (pos.x + j % 3) * dir;
                pos.z = (pos.z - j / 3) * dir; //反対向き

                GameObject obj = Instantiate(prefabUnitTile, pos, Quaternion.identity);
                unitTiles[i].Add(obj);

                obj.SetActive(false);
            }
        }

        //敵陣設定
        enemyLines = new List<int>[PlayerMax];
        for (int i = 0; i < PlayerMax; i++)
        {
            enemyLines[i] = new List<int>();
            int rangemin = 0;
            if (0 == i)
            {
                rangemin = boardHeight - EnemyLine;
            }

            for (int j = 0; j < EnemyLine; j++)
            {
                enemyLines[i].Add(rangemin + j);
            }
        }

        //TurnChangeからはじめる場合-1
        nowPlayer = -1;

        //初回モード
        nowMode = Mode.None;
        nextMode = Mode.TurnChange;
    }

    // Update is called once per frame
    void Update()
    {
        if (serverBoard.HasServerState)
        {
            if (serverBoard.CanSelectMove && nowMode == Mode.Select) selectMode();
            if (nextMode != Mode.None) { nowMode = nextMode; nextMode = Mode.None; }
            return;
        }
        if (Mode.Start == nowMode)
        {
            startMode();
        }
        else if (Mode.Select == nowMode)
        {
            selectMode();
        }
        else if (Mode.TurnChange == nowMode)
        {
            turnChangeMode();
        }

        //モード変更
        if (Mode.None != nextMode)
        {
            nowMode = nextMode;
            nextMode = Mode.None;
        }
    }



    //選択時
    void setSelectCursors(UnitController unit = null, bool playerunit = true)
    {
        //カーソル削除
        foreach (var item in cursors)
        {
            Destroy(item);
        }
        cursors.Clear();

        //選択中のユニットがあれば選択解除
        if (selectUnit)
        {
            selectUnit.Select(false);
            selectUnit = null;
        }

        //ユニット情報がなければ修了
        if (!unit) return;

        //移動可能範囲取得
        List<Vector2Int> movabletiles = getMovableTiles(unit);
        movableTiles.Clear();

        foreach (var item in movabletiles)
        {
            movableTiles.Add(tiles[item], item);
            //カーソル生成
            Vector3 pos = tiles[item].transform.position;
            pos.y += 0.51f;
            GameObject cursor = Instantiate(prefabCursor, pos, Quaternion.identity);
            cursors.Add(cursor);


        }



        //新しいユニットを選択
        if (playerunit)
        {
            unit.Select();
            selectUnit = unit;
        }
    }

    // 移動先と成りの選択を確定し、盤面変更は接近完了後に行う。
    Mode moveUnit(UnitController unit, Vector2Int tileindex)
    {
        pendingMoveUnit = unit;
        pendingMoveDestination = tileindex;
        bool isDrop = unit.FieldStatus == FieldStatus.Captured;
        pendingPlayerMove = isCpu ? null : new LegalMove
        {
            from = isDrop ? "00" : ToShogiSquare(unit.Pos),
            to = ToShogiSquare(tileindex),
            promote = false,
            drop_species = isDrop ? GetDropSpecies(unit.UnitType) : null
        };
        if (serverBoard.HasServerState)
        {
            bool normalAllowed = serverBoard.IsLegal(pendingPlayerMove, false);
            bool promotionAllowed = serverBoard.IsLegal(pendingPlayerMove, true);
            if (!normalAllowed && !promotionAllowed) { pendingMoveUnit = null; pendingPlayerMove = null; return Mode.Select; }
            if (normalAllowed && promotionAllowed)
            {
                textResultInfo.text = "成りますか？";
                buttonEvolutionApply.gameObject.SetActive(true);
                buttonEvolutionCancel.gameObject.SetActive(true);
                return Mode.WaitEvolution;
            }
            return BeginPendingMove(promotionAllowed);
        }
        bool canPromote = !isDrop && unit.isEvolution()
            && (enemyLines[nowPlayer].Contains(tileindex.y) || enemyLines[nowPlayer].Contains(unit.Pos.y));
        // 最奥の歩・香、最奥二段の桂は不成では次に動けない。
        int ranksRemaining = nowPlayer == 0 ? boardHeight - 1 - tileindex.y : tileindex.y;
        bool mustPromote = canPromote && (isCpu
            || ((unit.UnitType == UnitType.Hu || unit.UnitType == UnitType.Kyousha) && ranksRemaining == 0)
            || (unit.UnitType == UnitType.Keima && ranksRemaining <= 1));
        if (canPromote && !mustPromote)
        {
            textResultInfo.text = "成りますか？";
            buttonEvolutionApply.gameObject.SetActive(true);
            buttonEvolutionCancel.gameObject.SetActive(true);
            return Mode.WaitEvolution;
        }
        return BeginPendingMove(mustPromote);
    }

    Mode BeginPendingMove(bool promote)
    {
        UnitController unit = pendingMoveUnit;
        Vector2Int destination = pendingMoveDestination;
        pendingMoveUnit = null; // 二重クリックで同じ着手を開始させない。
        buttonEvolutionApply.gameObject.SetActive(false);
        buttonEvolutionCancel.gameObject.SetActive(false);
        textResultInfo.text = "";
        setSelectCursors();
        if (serverBoard.HasServerState)
        {
            pendingPlayerMove.promote = promote;
            serverBoard.RequestMove(pendingPlayerMove);
            pendingPlayerMove = null;
            return Mode.Select;
        }
        bool capture = units[destination.x, destination.y] != null;
        if ((capture || promote) && pieceCameraAnimator && pieceCameraAnimator.IsAvailable)
        {
            StartCoroutine(PlayPendingMove(unit, destination, promote));
            return Mode.Animating;
        }
        CommitMove(unit, destination, promote);
        return Mode.TurnChange;
    }

    IEnumerator PlayPendingMove(UnitController unit, Vector2Int destination, bool promote)
    {
        Vector3 focus = tiles[destination].transform.position;
        focus.y = UnitController.UnSelectUnitY;
        pieceCameraAnimator.FocusAt(focus, true);
        // 呼び出し元のモード更新を待つ。カメラ未設定時も進行を止めない。
        yield return null;
        while (pieceCameraAnimator && pieceCameraAnimator.IsPlaying && !pieceCameraAnimator.HasReachedFocus)
            yield return null;
        CommitMove(unit, destination, promote);
        if (pieceCameraAnimator) pieceCameraAnimator.CompleteAction();
        while (pieceCameraAnimator && pieceCameraAnimator.IsPlaying)
            yield return null;
        nextMode = Mode.TurnChange;
    }

    void CommitMove(UnitController unit, Vector2Int destination, bool promote)
    {
        Vector2Int origin = unit.Pos;
        bool isDrop = unit.FieldStatus == FieldStatus.Captured;
        captureUnit(nowPlayer, destination);
        unit.Move(tiles[destination], destination);
        if (isDrop) captureUnits.Remove(unit);
        else units[origin.x, origin.y] = null;
        units[destination.x, destination.y] = unit;
        unit.FieldStatus = FieldStatus.OnBoard;
        if (promote) unit.Evolution();
        alignCaptureUnits(nowPlayer);
        SetMoveCount(turnCount + 1);
        SendPendingPlayerMove(promote);
    }

    void SetMoveCount(int ply)
    {
        turnCount = ply;
        if (textTurnNumber) textTurnNumber.text = $"{turnCount}手";
    }

    // 配列は左下が(0, 0)。将棋の筋段は右上が11、左下が99。
    public static string ToShogiSquare(Vector2Int position)
    {
        if (position.x < 0 || position.x >= 9 || position.y < 0 || position.y >= 9)
            throw new ArgumentOutOfRangeException(nameof(position));
        return $"{9 - position.x}{9 - position.y}";
    }

    public static string GetDropSpecies(UnitType type)
    {
        switch (type)
        {
            case UnitType.Hu: return "FU";
            case UnitType.Kaku: return "KA";
            case UnitType.Hisya: return "HI";
            case UnitType.Kyousha: return "KY";
            case UnitType.Keima: return "KE";
            case UnitType.Gin: return "GI";
            case UnitType.Kin: return "KI";
            default: throw new ArgumentOutOfRangeException(nameof(type), type, "打てない駒です。");
        }
    }

    void SendPendingPlayerMove(bool promote)
    {
        if (pendingPlayerMove == null) return;
        LegalMove move = pendingPlayerMove;
        pendingPlayerMove = null;
        move.promote = promote;

        if (UnityWebSocketClient.Instance)
            UnityWebSocketClient.Instance.SendMoveRequest(move);
        else
            Debug.LogError("GameSceneDirector: UnityWebSocketClientが見つからないため、着手を送信できません。", this);
    }

    //移動可能範囲の取得
    List<Vector2Int> getMovableTiles(UnitController unit)
    {
        if (serverBoard.HasServerState)
            return serverBoard.GetDestinations(unit);
        //通常移動範囲
        List<Vector2Int> ret = unit.GetMovableTiles(units);

        //王手されてしまうかチェック
        UnitController[,] copyunits = GetCopyArray(units);
        if (FieldStatus.OnBoard == unit.FieldStatus)
        {
            copyunits[unit.Pos.x, unit.Pos.y] = null;
        }
        int outecount = GetOuteUnits(copyunits, unit.Player).Count;

        //王手を回避できる場所を返す
        if (0 < outecount)
        {
            ret = new List<Vector2Int>();
            List<Vector2Int> movabletiles = unit.GetMovableTiles(units);
            foreach (var item in movabletiles)
            {
                //移動した状態を作る
                UnitController[,] copyunits2 = GetCopyArray(copyunits);
                copyunits2[item.x, item.y] = unit;
                outecount = GetOuteUnits(copyunits2, unit.Player, false).Count;
                if (1 > outecount) ret.Add(item);
            }
        }

        return ret;
    }

    //ターン開始
    void startMode()
    {
        //勝敗がついていなければ通常モード
        nextMode = Mode.Select;

        //Info更新
        textTurnInfo.text = "" + (nowPlayer + 1) + "Pの番です";
        textResultInfo.text = "";

        //勝敗チェック

        //王手しているユニット判定
        List<UnitController> outeunits = GetOuteUnits(units, nowPlayer);
        bool isoute = 0 < outeunits.Count;
        if (isoute)
        {
            textResultInfo.text = "王手！";
        }

        int movablecount = 0;
        foreach (var item in getUnits(nowPlayer))
        {
            movablecount += getMovableTiles(item).Count;
        }

        //動かせないとき
        if (1 > movablecount)
        {
            textResultInfo.text = "移動できません";
            if (isoute)
            {
                textResultInfo.text = "詰み\n" + (GetNextPlayer(nowPlayer) + 1) + "Pの勝ち";
            }
            nextMode = Mode.Result;
        }

        //次が結果表示なら
        if (Mode.Result == nextMode)
        {
            textTurnInfo.text = "";
            buttonRematch.gameObject.SetActive(true);
            buttonTitle.gameObject.SetActive(true);
        }

    }

    //ユニットとタイル選択
    void selectMode()
    {
        //モード選択画面の表示中は、背後の盤や駒をクリックさせない
        if (!ModeSelectionManager.IsWorldInteractionAllowed) return;
        if (pieceCameraAnimator && pieceCameraAnimator.IsPlaying) return;

        GameObject tile = null;
        UnitController unit = null;

        if (Input.GetMouseButtonUp(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition); //奥のタイルの情報も欲しいので

            foreach (RaycastHit hit in Physics.RaycastAll(ray))
            {
                UnitController hitunit = hit.transform.GetComponent<UnitController>();

                //持ち駒
                if (hitunit && FieldStatus.Captured == hitunit.FieldStatus)
                {
                    unit = hitunit;
                }
                //タイル選択と上に乗っているユニット
                else if (tiles.ContainsValue(hit.transform.gameObject))
                {
                    tile = hit.transform.gameObject;
                    //タイルからユニットを探す
                    foreach (var item in tiles)
                    {
                        if (item.Value == tile)
                        {
                            unit = units[item.Key.x, item.Key.y];
                        }
                    }
                    break;
                }
            }
        }

        //なにも選択されていなければ処理をしない
        if (null == tile && null == unit) return;

        //移動先選択
        if (tile && selectUnit && movableTiles.ContainsKey(tile))
        {
            nextMode = moveUnit(selectUnit, movableTiles[tile]);

        }

        //ユニット選択
        else if (unit)
        {
            bool isplayer = nowPlayer == unit.Player;
            setSelectCursors(unit, isplayer);
        }
    }

    //ターン変更
    void turnChangeMode()
    {
        //ボタンとカーソルのリセット
        setSelectCursors();
        buttonEvolutionApply.gameObject.SetActive(false);
        buttonEvolutionCancel.gameObject.SetActive(false);

        //CPU状態解除
        isCpu = false;

        //次のプレイヤーへ
        nowPlayer = GetNextPlayer(nowPlayer);

        nextMode = Mode.Start;
    }

    //次のプレイヤー番号を返す
    public static int GetNextPlayer(int player)
    {
        int next = player + 1;
        if (PlayerMax <= next) next = 0;

        return next;
    }

    //ユニットを持ち駒にする
    void captureUnit(int player, Vector2Int tileindex)
    {
        UnitController unit = units[tileindex.x, tileindex.y];
        if (!unit) return;
        unit.Capture(player);
        captureUnits.Add(unit);
        units[tileindex.x, tileindex.y] = null;
    }

    //持ち駒を並べる
    void alignCaptureUnits(int player)
    {
        //所持個数を一旦非表示に
        foreach (var item in unitTiles[player])
        {
            item.SetActive(false);
        }

        //ユニットごとに分ける
        Dictionary<UnitType, List<UnitController>> typeunits
        = new Dictionary<UnitType, List<UnitController>>();

        foreach (var item in captureUnits)
        {
            if (player != item.Player) continue;
            typeunits.TryAdd(item.UnitType, new List<UnitController>());
            typeunits[item.UnitType].Add(item);
        }

        //タイプごとに並べて一番上だけ表示
        int tilecount = 0;
        foreach (var item in typeunits)
        {
            if (1 > item.Value.Count) continue;

            //置く場所
            GameObject tile = unitTiles[player][tilecount++]; //次でプラス表示したいので

            //非表示にしていたタイルを表示する
            tile.SetActive(true);

            //所持個数の表示
            tile.transform.GetChild(0).gameObject.GetComponent<TextMeshPro>().text
            = "" + item.Value.Count;

            //同じ種類の持ち駒を並べる
            for (int i = 0; i < item.Value.Count; i++)
            {
                //リスト内のユニットを表示
                GameObject unit = item.Value[i].gameObject;
                //置く場所
                Vector3 pos = tile.transform.position;
                //一旦ユニットを移動して表示する
                unit.SetActive(true);
                unit.transform.position = pos;
                //1個目以外は非表示
                if (0 < i) unit.SetActive(false);
            }
        }

    }

    //指定した配列をコピーして返す
    public static UnitController[,] GetCopyArray(UnitController[,] ary)
    {
        UnitController[,] ret = new UnitController[ary.GetLength(0), ary.GetLength(1)];
        Array.Copy(ary, ret, ary.Length);
        return ret;
    }

    //指定された配置で王手しているユニットを返す
    public static List<UnitController> GetOuteUnits(UnitController[,] units, int player, bool checkotherunit = true)
    {
        List<UnitController> ret = new List<UnitController>();
        foreach (var unit in units)
        {
            if (!unit || player == unit.Player) continue;

            //ユニットの移動可能範囲
            List<Vector2Int> movabletiles = unit.GetMovableTiles(units, checkotherunit);

            foreach (var tile in movabletiles)
            {
                if (!units[tile.x, tile.y]) continue;

                if (UnitType.Gyoku == units[tile.x, tile.y].UnitType)
                {
                    ret.Add(unit);
                }
            }
        }
        return ret;
    }

    //成るボタン
    public void OnClickEvolutionApply()
    {
        if (!pendingMoveUnit || nowMode != Mode.WaitEvolution) return;
        nextMode = BeginPendingMove(true);
    }

    //成らないボタン
    public void OnClickEvolutionCancel()
    {
        if (!pendingMoveUnit || nowMode != Mode.WaitEvolution) return;
        nextMode = BeginPendingMove(false);
    }

    //指定されたプレイヤー番号の全ユニットを取得する
    List<UnitController> getUnits(int player)
    {
        List<UnitController> ret = new List<UnitController>();

        //全ユニットのリストを作成する
        List<UnitController> allunits = new List<UnitController>(captureUnits);
        allunits.AddRange(units);
        foreach (var item in allunits)
        {
            if (!item || player != item.Player) continue;
            ret.Add(item);
        }
        return ret;
    }

    //TODO 検討：ゲーム開始前後の仕様
    //リザルト,再戦
    public void OnClickRematch()
    {
        SceneManager.LoadScene("MainGame");
    }

    //タイトルへ
    public void OnClickTitle()
    {
        SceneManager.LoadScene("TitleScene");
    }
}
