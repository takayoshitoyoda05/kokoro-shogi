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

public class TitleSceneDirector : MonoBehaviour
{
    // タイルのプレハブ
    [SerializeField] GameObject prefabTile;

    // ユニットのプレハブ
    [SerializeField] List<GameObject> prefabUnits;

    // 初期配置
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

    // Start is called before the first frame update
    void Start()
    {
        // ボードサイズ
        int boardWidth = boardSetting.GetLength(0);
        int boardHeight = boardSetting.GetLength(1);

        for (int i = 0; i < boardWidth; i++)
        {
            for (int j = 0; j < boardHeight; j++)
            {
                // タイルとユニットのポジション
                float x = i - boardWidth / 2;
                float y = j - boardHeight / 2;

                // ポジション
                Vector3 pos = new Vector3(x, 0, y);

                // タイルのインデックス
                Vector2Int tileindex = new Vector2Int(i, j);

                // タイル作成
                GameObject tile = Instantiate(prefabTile, pos, Quaternion.identity);

                // ユニット作成
                int type = boardSetting[i, j] % 10;
                int player = boardSetting[i, j] / 10;

                if (0 == type) continue;

                // 初期化
                pos.y = 0.7f;

                GameObject prefab = prefabUnits[type - 1];
                GameObject unit = Instantiate(prefab, pos, Quaternion.Euler(90, player * 180, 0));
                unit.AddComponent<Rigidbody>();

                UnitController unitctrl = unit.AddComponent<UnitController>();
                unitctrl.Init(player, type, tile, tileindex);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void OnClickPvP()
    {
        // GameSceneDirector.PlayerCount = 2;
        SceneManager.LoadScene("MainGame");
    }

    public void OnClickPvE()
    {
        // GameSceneDirector.PlayerCount = 1;
        SceneManager.LoadScene("MainGame");
    }

    public void OnClickEvE()
    {
        // GameSceneDirector.PlayerCount = 0;
        SceneManager.LoadScene("MainGame");
    }
}
