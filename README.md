根據該期刊與 gemini pro 3 來嘗試實現 HGA 解 MSTSP
https://ieeexplore.ieee.org/document/10981731

---

為了讓 Javascript 可以使用多個執行緒來運算，有額外啟動 worker，local 環境需要額外啟動 web server，才可以解決 CORS 問題

### local server

```shell
python3 -m http.server
```

http://localhost:8000