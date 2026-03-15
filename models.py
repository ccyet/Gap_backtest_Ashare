from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisParams:
    data_source_type: str
    db_path: str
    table_name: str | None
    column_overrides: dict[str, str]
    excel_sheet_name: str | None
    start_date: str
    end_date: str
    stock_codes: tuple[str, ...]
    gap_direction: str
    gap_pct: float
    max_gap_filter_pct: float
    use_ma_filter: bool
    fast_ma_period: int
    slow_ma_period: int
    time_stop_days: int
    time_stop_target_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    enable_take_profit: bool
    enable_profit_drawdown_exit: bool
    profit_drawdown_pct: float
    enable_ma_exit: bool
    exit_ma_period: int
    ma_exit_batches: int
    buy_cost_pct: float
    sell_cost_pct: float
    time_exit_mode: str

    @property
    def gap_ratio(self) -> float:
        return self.gap_pct / 100.0

    @property
    def time_stop_target_ratio(self) -> float:
        return self.time_stop_target_pct / 100.0

    @property
    def stop_loss_ratio(self) -> float:
        return self.stop_loss_pct / 100.0

    @property
    def take_profit_ratio(self) -> float:
        return self.take_profit_pct / 100.0

    @property
    def profit_drawdown_ratio(self) -> float:
        return self.profit_drawdown_pct / 100.0

    @property
    def buy_cost_ratio(self) -> float:
        return self.buy_cost_pct / 100.0

    @property
    def sell_cost_ratio(self) -> float:
        return self.sell_cost_pct / 100.0

    @property
    def required_lookback_days(self) -> int:
        lookback = 5
        if self.use_ma_filter:
            lookback = max(lookback, max(self.fast_ma_period, self.slow_ma_period) + 5)
        if self.enable_ma_exit:
            lookback = max(lookback, self.exit_ma_period + 5)
        return lookback

    @property
    def required_lookahead_days(self) -> int:
        return self.time_stop_days + 5


def normalize_stock_codes(raw_text: str) -> tuple[str, ...]:
    if not raw_text.strip():
        return ()
    separators = [",", "，", "\n", "\t", " "]
    normalized = raw_text
    for separator in separators[1:]:
        normalized = normalized.replace(separator, separators[0])
    parts = [item.strip().upper() for item in normalized.split(separators[0])]
    return tuple(code for code in parts if code)


def normalize_column_overrides(raw_values: dict[str, str]) -> dict[str, str]:
    return {key: value.strip() for key, value in raw_values.items() if value.strip()}


def validate_params(params: AnalysisParams) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if params.data_source_type not in {"sqlite", "file"}:
        errors.append("数据来源类型不合法。")

    if params.data_source_type == "sqlite" and not params.db_path.strip():
        errors.append("请填写 SQLite 数据库文件路径。")

    if params.gap_direction not in {"up", "down"}:
        errors.append("跳空方向只能是向上或向下。")

    if params.time_exit_mode not in {"strict", "force_close"}:
        errors.append("时间到期处理方式不合法。")

    if params.gap_pct < 0:
        errors.append("跳空幅度不能为负数。")

    if params.max_gap_filter_pct < 0:
        errors.append("最大高开/低开过滤不能为负数。")

    if params.time_stop_days < 1:
        errors.append("最多持有天数必须大于等于 1。")

    if params.stop_loss_pct < 0:
        errors.append("止损比例不能为负数。")

    if params.take_profit_pct < 0:
        errors.append("止盈比例不能为负数。")

    if params.profit_drawdown_pct < 0:
        errors.append("盈利回撤比例不能为负数。")

    if params.enable_profit_drawdown_exit and params.profit_drawdown_pct > 100:
        warnings.append("盈利回撤比例大于 100%，通常会导致很难触发，请确认设置。")

    if params.enable_ma_exit and params.exit_ma_period < 1:
        errors.append("止盈参考均线周期必须大于等于 1。")

    if params.enable_ma_exit and not (2 <= params.ma_exit_batches <= 3):
        errors.append("均线离场分批数必须在 2 到 3 之间。")

    if params.buy_cost_pct < 0 or params.sell_cost_pct < 0:
        errors.append("买入成本和卖出成本都不能为负数。")

    if params.use_ma_filter:
        if params.fast_ma_period < 1 or params.slow_ma_period < 1:
            errors.append("均线周期必须大于等于 1。")
        if params.fast_ma_period > params.slow_ma_period:
            warnings.append("快线周期大于慢线周期是允许的，但请确认这符合您的研究习惯。")

    for canonical, label in (
        ("date", "日期列名"),
        ("stock_code", "股票代码列名"),
        ("open", "开盘价列名"),
        ("high", "最高价列名"),
        ("low", "最低价列名"),
        ("close", "收盘价列名"),
        ("volume", "成交量列名"),
    ):
        if canonical in params.column_overrides and not params.column_overrides[canonical].strip():
            errors.append(f"{label}不能为空。")

    if params.stop_loss_pct >= 100:
        warnings.append("止损比例大于等于 100%，这通常不符合常见交易设置。")

    if params.take_profit_pct >= 100:
        warnings.append("止盈比例大于等于 100%，请确认这是否为预期值。")

    if params.time_stop_target_pct < -100:
        warnings.append("到期最低目标涨幅过低，请确认是否填写正确。")

    if params.start_date > params.end_date:
        errors.append("开始日期不能晚于结束日期。")

    return errors, warnings
