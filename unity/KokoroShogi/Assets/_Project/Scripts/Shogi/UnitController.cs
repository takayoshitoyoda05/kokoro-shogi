using UnityEngine;
using System.Collections.Generic;

//駒のタイプ
public enum UnitType
{
    None = -1,
    Hu = 1,
    Kaku,
    Hisya,
    Kyousha,
    Keima,
    Gin,
    Kin,
    Gyoku,
    //成
    Tokin,
    Uma,
    Ryu,
    NariKyo,
    NariKei,
    NariGin,
}

//駒の場所
public enum FieldStatus
{
    OnBoard,
    Captured,
}

public class UnitController : MonoBehaviour
{
    //ユニットのプレイヤー番号
    public int Player;
    //ユニットの種類
    public UnitType UnitType, OldUnitType;
    //ユニットの場所
    public FieldStatus FieldStatus;

    //成テーブル
    Dictionary<UnitType, UnitType> evolutionTable = new Dictionary<UnitType, UnitType>()
    {
        {UnitType.Hu, UnitType.Tokin},
        {UnitType.Kaku, UnitType.Uma},
        {UnitType.Hisya, UnitType.Ryu},
        {UnitType.Kyousha, UnitType.NariKyo},
        {UnitType.Keima, UnitType.NariKei},
        {UnitType.Gin, UnitType.NariGin},
        {UnitType.Kin, UnitType.None},
        {UnitType.Gyoku, UnitType.None},
    };

    //成済みかどうか
    public bool isEvolution;

    //ユニット選択/非選択のy座標
    public const float SelectUnitY = 1.5f;
    public const float UnSelectUnitY = 0.7f;

    //置いている場所
    public Vector2Int Pos;

    //選択される前のy座標
    float oldPosY;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    //初期設定
    public void Init(int player, int unittype, GameObject tile, Vector2Int pos)
    {
        Player = player;
        UnitType = (UnitType)unittype;
        //取られたとき元に戻るように
        OldUnitType = (UnitType)unittype;
        //場所の初期値
        FieldStatus = FieldStatus.OnBoard;
        //角度と場所
        transform.eulerAngles = getDefaultAngles(player);
        Move(tile, pos);
    }

    //指定されたプレイヤーあ番号の角度を返す
    Vector3 getDefaultAngles(int player)
    {
        return new Vector3(90, player * 180, 0);
    }

    //移動処理
    public void Move(GameObject tile, Vector2Int tileindex)
    {
        //新しい場所に移動する
        Vector3 pos = tile.transform.position;
        pos.y = UnSelectUnitY;
        transform.position = pos;
        //インデックス更新
        Pos = tileindex;
    }

    //選択時の処理
    public void Select(bool select = true)
    {
        Vector3 pos = transform.position;
        bool iskinematic = select;

        if (select)
        {
            oldPosY = pos.y;
            pos.y = SelectUnitY;
        }
        else
        {
            pos.y = UnSelectUnitY; //駒を持ち上げたとき重力無し

            //持ち駒は重力あり
            if(FieldStatus.Captured == FieldStatus)
            {
                pos.y = oldPosY;
                iskinematic = true;
            }
        }
    

        GetComponent<Rigidbody>().isKinematic = iskinematic;
        transform.position = pos;
    }

    //移動可能範囲の取得
    public List<Vector2Int> GetMovableTiles(UnitController[,] units, bool checkotherunit = true)
    {
        List<Vector2Int> ret = new List<Vector2Int>();

        ret = getMovableTiles(units, UnitType.Hu);

        return ret;
    }

    //もととなる移動可能範囲制限
    List<Vector2Int> getMovableTiles(UnitController[,] units, UnitType unittype)
    {
        List<Vector2Int> ret = new List<Vector2Int>();

        //歩
        if(UnitType.Hu == unittype)
        {
            //向き
            int dir = (0 == Player) ? 1 : -1;

            //前方1マス
            List<Vector2Int> vec = new List<Vector2Int>()
            {
                new Vector2Int(0, 1 * dir)
            };

            //実際のフィールドを調べる
            foreach(var item in vec)
            {
                Vector2Int checkpos = Pos + item;
                if(!isCheckable(units, checkpos)||isFriendlyUnit(units[checkpos.x, checkpos.y])) //配列オーバーor仲間のユニットがあれば戻る
                {
                    continue;
                }

                ret.Add(checkpos);
            }
        }

        return ret;
    }

    //配列オーバーかどうか
    bool isCheckable(UnitController[,] ary, Vector2Int idx)
    {
        //配列オーバーの状態
        if(    idx.x < 0 || ary.GetLength(0) <= idx.x 
            || idx.y < 0 || ary.GetLength(1) <= idx.y)
        {
            return false;
        }
        return true;
    }

    //仲間のユニットかどうか
    bool isFriendlyUnit(UnitController unit)
    {
        if(unit && Player == unit.Player) return true;
        return false;
    }

    //キャプチャされたとき
    public void Capture(int player)
    {
        Player = player;
        FieldStatus = FieldStatus.Captured;
        Evolution(false);
        GetComponent<Rigidbody>().isKinematic = true;
    }

    //成
    public void Evolution(bool evolution = true)
    {
        Vector3 angle = transform.eulerAngles;

        //成
        if(evolution && UnitType.None != evolutionTable[UnitType])
        {
            UnitType = evolutionTable[UnitType];
            angle.x = 270;
            angle.y = (0 == Player) ? 180 : 0;
            angle.z = 0;
            transform.eulerAngles = angle;
        }
        else
        {
            UnitType = OldUnitType;
            transform.eulerAngles = getDefaultAngles(Player);
        }

        isEvolution = evolution;
    }

}
