# Gap_test（A股跳空回测）

一个基于 **Streamlit + Pandas** 的 A 股跳空策略回测工具。

当前项目重点支持：
- **离线数据更新**（AKShare 仅用于下载，回测层不直接依赖）
- **本地 Parquet 数据源**
- **单账户单持仓回测**
- **分批止盈（2~3 批）与 fill 级别交易明细**

---

## 1. 项目定位与核心原则

### 1.1 定位
本工具用于在指定时间区间内：
1. 扫描“向上/向下跳空”信号；
2. 按统一规则模拟买卖；
3. 输出交易明细、日统计、净值曲线及可导出结果。

### 1.2 核心原则（非常重要）
- **数据下载与回测解耦**：
  - `akshare` 只在数据提供层/更新脚本中出现；
  - 回测核心（策略规则、分析汇总）不直接调用 akshare。
- **统一标准代码**：股票代码统一为 `000001.SZ` / `600519.SH` 这类带后缀格式。
- **本地优先**：回测读取本地标准化数据，不依赖在线接口实时可用性。

---

## 2. 快速开始

## 2.1 安装依赖

```bash
pip install -r requirements.txt
```

## 2.2 启动 Web 界面

```bash
streamlit run app.py
```

启动后你可以：
- 在页面上方“**本地行情更新（离线下载）**”面板更新本地数据；
- 在“参数设置”里配置策略并直接回测。

---

## 3. 数据流（建议先理解）

### 3.1 离线更新流程
1. UI 或命令行调用 `scripts/update_data.py`
2. `data/providers/akshare_provider.py` 拉取日线
   - 下载源优先级：**新浪 → 腾讯 → 东财**（逐级 fallback）
3. 清洗并标准化（日期/数值/代码）
4. 写入本地 parquet：
   - `data/market/daily/{adjust}/{symbol}.parquet`
5. 追加更新日志：
   - `data/market/metadata/update_log.parquet`
6.（可选）导出 Excel：
   - `data/market/exports/{adjust}/{symbol}.xlsx`

### 3.2 回测读取流程
1. 页面参数提交
2. 回测逻辑读取本地数据（及上传文件/SQLite 方式）
3. `rules.py` 执行交易模拟
4. `analyzer.py` 汇总结果并输出报表

---

## 4. 目录结构（关键部分）

```text
config/
  data_source.yaml                  # 数据源配置（local_parquet 等）

data/
  providers/
    akshare_provider.py             # 仅此层封装 akshare
  services/
    local_data_service.py           # 本地 parquet 读取接口
  market/
    daily/{adjust}/{symbol}.parquet # 本地日线
    metadata/
      symbols.parquet               # 股票列表
      update_log.parquet            # 更新日志
    exports/{adjust}/{symbol}.xlsx  # 可选导出

scripts/
  update_data.py                    # 离线更新脚本

app.py                              # Streamlit 页面
rules.py                            # 单笔交易模拟与退出规则
analyzer.py                         # 候选交易、策略交易、净值汇总
models.py                           # 参数模型与校验
```

---

## 5. 配置说明

`config/data_source.yaml`：

```yaml
data_source: local_parquet
local_data_root: data/market/daily
default_adjust: qfq
```

含义：
- `data_source`：当前使用本地 parquet
- `local_data_root`：本地行情根目录
- `default_adjust`：默认复权（如 qfq）

---

## 6. 离线更新用法

## 6.1 在 Web 界面更新（推荐）
在页面“本地行情更新（离线下载）”中填写：
- 股票代码（可选）
- 起止日期
- 复权类型
- 是否刷新股票列表
- 是否导出 Excel

然后点击“开始更新本地数据”。

## 6.2 命令行更新

```bash
python scripts/update_data.py --start-date 2024-01-01 --end-date 2024-12-31 --adjust qfq
```

常用参数：
- `--symbols`：逗号分隔，如 `000001.SZ,600519.SH`
- `--refresh-symbols`：先刷新 `symbols.parquet`
- `--export-excel`：更新后额外导出 Excel

---

## 7. 回测规则概览

## 7.1 交易框架
- 单账户、单持仓
- 持仓期间不重复开仓
- 平仓后才释放仓位
- 支持 long / short 的研究型镜像回测；short 仅用于研究，不代表 A 股可直接实盘融券交易。

## 7.2 退出优先级（日内）
1. 全仓止损
2. 分批退出（按 `priority`）
3. 旧版整笔退出（仅在未启用分批时）
4. 时间退出（`holding_days >= N` 后持续检查）
5. 数据结束处理（`strict` / `force_close`）

## 7.3 分批止盈（partial exit）
- 支持 2~3 批
- 每批可独立配置：
  - `weight_pct`（仓位比例）
  - `mode`（`fixed_tp` / `ma_exit` / `profit_drawdown`）
  - `priority`
- 同一天可触发多批，按 priority 从小到大执行
- 返回 fill 级别明细：`fills`、`fill_count`、`fill_detail_json`

## 7.4 profit_drawdown 定义
采用“**整笔交易总利润回撤**”而非单纯价格峰值回撤：
- `realized_profit`：已成交批次锁定的累计利润
- `unrealized_profit`：剩余仓位按当前价格估算的浮动利润
- `total_profit_now = realized_profit + unrealized_profit`
- `peak_total_profit`：持仓以来出现过的最高总利润
- `profit_drawdown = (peak_total_profit - total_profit_now) / peak_total_profit`

触发条件：
1. `peak_total_profit` 先达到最小激活浮盈门槛
2. 总利润回撤比例达到阈值

这意味着：
- 第一批止盈后，后续批次的 `profit_drawdown` 会把已锁定利润一起纳入计算
- 第二/第三批退出依据是“整笔交易总利润回撤”，不是“最高价回撤”
- whole-position 与 partial `profit_drawdown` 现在使用同一套总利润语义

---

## 8. 结果输出

主要输出三类：
1. **交易明细**（每笔交易）
2. **按开仓日汇总**（日统计）
3. **净值曲线**

其中交易明细包含：
- `fills`（分批成交列表）
- `fill_count`
- `fill_detail_json`
- 归一化加权 `sell_price`

---

## 9. 测试

运行全部测试：

```bash
pytest -q
```

当前测试覆盖重点：
- 分批退出与优先级
- 总利润回撤（含先止盈再回撤的分批场景）
- 时间退出行为
- strict / force_close
- 本地 parquet 读取
- AKShare 源 fallback（新浪 → 腾讯 → 东财）

---

## 10. 常见问题

### Q1：为什么更新会失败？
常见是网络或代理问题。系统会自动按“新浪→腾讯→东财”递补，三源都失败时会在日志中输出合并错误信息。

### Q2：为什么回测时看不到最新数据？
请确认：
- 更新脚本执行成功；
- parquet 文件已写入 `data/market/daily/{adjust}`；
- 回测日期范围包含已下载区间。

### Q3：为什么代码必须带后缀？
为了避免跨市场 6 位代码碰撞，项目统一使用 `xxxxxx.SZ/SH/BJ` 标准格式。

---

## 11. 开发提示

- 如果你要改下载逻辑，请只改 `data/providers/` 和 `scripts/update_data.py`。
- 如果你要改策略逻辑，请集中在 `rules.py` / `analyzer.py` / `models.py`。
- 任何情况下，都尽量保持“下载与回测解耦”。


---

## 12. 主题配置（含暗色主题）

本项目不在 `app.py` 内动态切主题。

如需暗色主题，请在项目根目录新建：` .streamlit/config.toml `，示例：

```toml
[theme]
base="dark"
primaryColor="#4f8bf9"
backgroundColor="#0e1117"
secondaryBackgroundColor="#262730"
textColor="#fafafa"
```

保存后重启 `streamlit run app.py` 生效。
