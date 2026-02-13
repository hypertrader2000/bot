import sqlite3

DB_PATH = "data/bot.sqlite3"

RESET = {
    "sim_quote": "1000",
    "sim_base": "0",
    "position_json": "{}",
    "last_trade_ts": "0",
    "last_any_trade_ms": "0",
}

def main():
    con = sqlite3.connect(DB_PATH)

    # Discover columns in `state`
    cols = con.execute("PRAGMA table_info(state);").fetchall()
    # cols rows: (cid, name, type, notnull, dflt_value, pk)
    col_names = [c[1] for c in cols]

    # Find the key column
    key_col = None
    for candidate in ("key", "k", "name"):
        if candidate in col_names:
            key_col = candidate
            break
    if key_col is None:
        # fallback: first TEXT-ish column or first column
        key_col = col_names[0]

    # Find the value column (the one that is NOT the key)
    value_col = None
    for c in col_names:
        if c != key_col:
            value_col = c
            break
    if value_col is None:
        raise RuntimeError(f"Could not determine value column from state table columns: {col_names}")

    print(f"Detected state schema: key_col='{key_col}', value_col='{value_col}'")

    # Make sure we can uniquely identify keys
    # Insert missing keys, update existing
    for k, v in RESET.items():
        # Does key exist?
        row = con.execute(f"SELECT 1 FROM state WHERE {key_col}=? LIMIT 1", (k,)).fetchone()
        if row:
            con.execute(f"UPDATE state SET {value_col}=? WHERE {key_col}=?", (v, k))
        else:
            con.execute(f"INSERT INTO state ({key_col}, {value_col}) VALUES (?, ?)", (k, v))

    con.commit()
    con.close()
    print("✅ SIM RESET COMPLETE (sim_quote=1000, sim_base=0, position cleared)")

if __name__ == "__main__":
    main()
