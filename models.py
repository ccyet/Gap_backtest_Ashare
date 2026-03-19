from __future__ import annotations

from dataclasses import dataclass, field, replace


GAP_ENTRY_MODES = ("strict_break", "open_vs_prev_close_threshold")
ENTRY_FACTORS = ("gap", "trend_breakout", "volatility_contraction_breakout")
SCAN_METRICS = (
    "signal_count",
    "closed_trade_candidates",
    "executed_trades",
    "strategy_win_rate_pct",
    "total_return_pct",
    "max_drawdown_pct",
    "final_net_value",
    "avg_holding_days",
    "profit_risk_ratio",
    "trade_return_volatility_pct",
)
SCAN_FIELD_CASTERS: dict[str, type[int] | type[float]] = {
    "gap_pct": float,
    "max_gap_filter_pct": float,
    "trend_breakout_lookback": int,
    "vcb_range_lookback": int,
    "vcb_breakout_lookback": int,
    "time_stop_days": int,
    "time_stop_target_pct": float,
    "stop_loss_pct": float,
    "take_profit_pct": float,
    "profit_drawdown_pct": float,
    "exit_ma_period": int,
    "buy_slippage_pct": float,
    "sell_slippage_pct": float,
}
FACTOR_SCAN_ELIGIBLE_FIELDS: dict[str, frozenset[str]] = {
    "gap": frozenset(
        {
            "gap_pct",
            "max_gap_filter_pct",
            "time_stop_days",
            "time_stop_target_pct",
            "stop_loss_pct",
            "take_profit_pct",
            "profit_drawdown_pct",
            "exit_ma_period",
            "buy_slippage_pct",
            "sell_slippage_pct",
        }
    ),
    "trend_breakout": frozenset(
        {
            "trend_breakout_lookback",
            "time_stop_days",
            "time_stop_target_pct",
            "stop_loss_pct",
            "take_profit_pct",
            "profit_drawdown_pct",
            "exit_ma_period",
            "buy_slippage_pct",
            "sell_slippage_pct",
        }
    ),
    "volatility_contraction_breakout": frozenset(
        {
            "vcb_range_lookback",
            "vcb_breakout_lookback",
            "time_stop_days",
            "time_stop_target_pct",
            "stop_loss_pct",
            "take_profit_pct",
            "profit_drawdown_pct",
            "exit_ma_period",
            "buy_slippage_pct",
            "sell_slippage_pct",
        }
    ),
}


EPSILON = 1e-12


@dataclass(frozen=True)
class PartialExitRule:
    enabled: bool
    weight_pct: float
    mode: str
    priority: int
    target_profit_pct: float | None = None
    ma_period: int | None = None
    drawdown_pct: float | None = None
    min_profit_to_activate_drawdown: float | None = None

    @property
    def weight_ratio(self) -> float:
        return self.weight_pct / 100.0

    @property
    def target_profit_ratio(self) -> float | None:
        if self.target_profit_pct is None:
            return None
        return self.target_profit_pct / 100.0

    @property
    def drawdown_ratio(self) -> float | None:
        if self.drawdown_pct is None:
            return None
        return self.drawdown_pct / 100.0

    @property
    def min_profit_to_activate_drawdown_ratio(self) -> float:
        threshold = self.min_profit_to_activate_drawdown
        if threshold is None:
            threshold = 5.0
        return threshold / 100.0


@dataclass(frozen=True)
class TradeFill:
    sell_date: str
    sell_price: float
    weight: float
    exit_type: str
    holding_days: int


@dataclass(frozen=True)
class ParamScanAxis:
    field_name: str
    values: tuple[int | float, ...]


@dataclass(frozen=True)
class ParamScanConfig:
    enabled: bool = False
    axes: tuple[ParamScanAxis, ...] = ()
    metric: str = "total_return_pct"
    max_combinations: int = 25

    @property
    def combination_count(self) -> int:
        if not self.enabled or not self.axes:
            return 0
        total = 1
        for axis in self.axes:
            total *= len(axis.values)
        return total


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
    partial_exit_enabled: bool
    partial_exit_count: int
    partial_exit_rules: tuple[PartialExitRule, ...]
    buy_cost_pct: float
    sell_cost_pct: float
    time_exit_mode: str
    entry_factor: str = "gap"
    gap_entry_mode: str = "strict_break"
    trend_breakout_lookback: int = 20
    vcb_range_lookback: int = 7
    vcb_breakout_lookback: int = 20
    buy_slippage_pct: float = 0.0
    sell_slippage_pct: float = 0.0
    local_data_root: str = "data/market/daily"
    adjust: str = "qfq"
    scan_config: ParamScanConfig = field(default_factory=ParamScanConfig)

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
    def buy_slippage_ratio(self) -> float:
        return self.buy_slippage_pct / 100.0

    @property
    def sell_slippage_ratio(self) -> float:
        return self.sell_slippage_pct / 100.0

    @property
    def required_lookback_days(self) -> int:
        lookback = 5
        if self.use_ma_filter:
            lookback = max(lookback, max(self.fast_ma_period, self.slow_ma_period) + 5)
        if self.enable_ma_exit:
            lookback = max(lookback, self.exit_ma_period + 5)
        if self.partial_exit_enabled:
            partial_ma_periods = [
                rule.ma_period
                for rule in self.partial_exit_rules
                if rule.enabled and rule.ma_period is not None
            ]
            if partial_ma_periods:
                lookback = max(lookback, max(partial_ma_periods) + 5)
        if self.entry_factor == "trend_breakout":
            lookback = max(lookback, self.trend_breakout_lookback + 5)
        if self.entry_factor == "volatility_contraction_breakout":
            lookback = max(
                lookback,
                max(self.vcb_range_lookback, self.vcb_breakout_lookback) + 5,
            )
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


def parse_scan_values(field_name: str, raw_text: str) -> tuple[int | float, ...]:
    caster = SCAN_FIELD_CASTERS[field_name]
    if not raw_text.strip():
        return ()
    normalized = raw_text
    for separator in ["，", "\n", "\t", " "]:
        normalized = normalized.replace(separator, ",")
    values: list[int | float] = []
    for chunk in normalized.split(","):
        text = chunk.strip()
        if not text:
            continue
        number = float(text)
        values.append(int(number) if caster is int else number)
    return tuple(values)


def apply_scan_overrides(
    params: AnalysisParams, overrides: dict[str, int | float]
) -> AnalysisParams:
    normalized: dict[str, int | float] = {}
    for field_name, value in overrides.items():
        caster = SCAN_FIELD_CASTERS[field_name]
        normalized[field_name] = int(value) if caster is int else float(value)
    return replace(params, **normalized)


def validate_params(params: AnalysisParams) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if params.data_source_type not in {"sqlite", "file", "local_parquet"}:
        errors.append("数据来源类型不合法。")

    if params.data_source_type == "sqlite" and not params.db_path.strip():
        errors.append("请填写 SQLite 数据库文件路径。")

    if params.gap_direction not in {"up", "down"}:
        errors.append("跳空方向只能是向上或向下。")

    if params.time_exit_mode not in {"strict", "force_close"}:
        errors.append("时间到期处理方式不合法。")

    if params.entry_factor not in ENTRY_FACTORS:
        errors.append("入场因子不合法。")

    if params.entry_factor == "gap":
        if params.gap_entry_mode not in GAP_ENTRY_MODES:
            errors.append("开仓信号模式不合法。")
    elif params.gap_entry_mode != "strict_break":
        errors.append("entry_factor 非 gap 时，不支持设置 gap_entry_mode。")

    if params.adjust not in {"qfq", "hfq"}:
        errors.append("复权方式只能为 qfq 或 hfq。")

    if params.gap_pct < 0:
        errors.append("跳空幅度不能为负数。")

    if params.max_gap_filter_pct < 0:
        errors.append("最大高开/低开过滤不能为负数。")

    if params.time_stop_days < 1:
        errors.append("最多持有天数必须大于等于 1。")

    if params.trend_breakout_lookback < 1:
        errors.append("trend_breakout_lookback 必须大于等于 1。")

    if params.vcb_range_lookback < 1:
        errors.append("vcb_range_lookback 必须大于等于 1。")

    if params.vcb_breakout_lookback < 1:
        errors.append("vcb_breakout_lookback 必须大于等于 1。")

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

    if params.partial_exit_count not in {2, 3}:
        errors.append("分批数量只能为 2 或 3。")

    if params.partial_exit_enabled:
        enabled_rules = [rule for rule in params.partial_exit_rules if rule.enabled]
        if len(enabled_rules) != params.partial_exit_count:
            errors.append("启用分批止盈时，启用规则数必须与分批数量一致。")

        total_weight = sum(rule.weight_pct for rule in enabled_rules)
        if abs(total_weight - 100.0) > EPSILON:
            errors.append("启用分批止盈时，规则仓位比例之和必须为 100%。")

        priorities = [rule.priority for rule in enabled_rules]
        if len(priorities) != len(set(priorities)):
            errors.append("启用分批止盈时，每批 priority 必须唯一。")

        valid_modes = {"fixed_tp", "ma_exit", "profit_drawdown"}
        for index, rule in enumerate(enabled_rules, start=1):
            if rule.mode not in valid_modes:
                errors.append(f"第 {index} 批退出方式不合法。")
                continue

            if rule.weight_pct < 0:
                errors.append(f"第 {index} 批仓位比例不能为负数。")

            if rule.mode == "fixed_tp":
                if rule.target_profit_pct is None:
                    errors.append(f"第 {index} 批 fixed_tp 必须填写目标收益。")
                elif rule.target_profit_pct < 0:
                    errors.append(f"第 {index} 批目标收益不能为负数。")

            if rule.mode == "ma_exit":
                if rule.ma_period is None:
                    errors.append(f"第 {index} 批 ma_exit 必须填写均线周期。")
                elif rule.ma_period < 1:
                    errors.append(f"第 {index} 批 ma_exit 均线周期必须大于等于 1。")

            if rule.mode == "profit_drawdown":
                if rule.drawdown_pct is None:
                    errors.append(f"第 {index} 批 profit_drawdown 必须填写回撤比例。")
                elif rule.drawdown_pct < 0:
                    errors.append(f"第 {index} 批回撤比例不能为负数。")

                if (
                    rule.min_profit_to_activate_drawdown is not None
                    and rule.min_profit_to_activate_drawdown < 0
                ):
                    errors.append(f"第 {index} 批最小浮盈激活门槛不能为负数。")

    if params.time_stop_days < 1:
        errors.append("启用时间退出时，time_stop_days 必须大于等于 1。")

    if params.time_exit_mode not in {"strict", "force_close"}:
        errors.append("time_exit_mode 只能为 strict 或 force_close。")

    if params.buy_cost_pct < 0 or params.sell_cost_pct < 0:
        errors.append("买入成本和卖出成本都不能为负数。")

    if params.buy_slippage_pct < 0 or params.sell_slippage_pct < 0:
        errors.append("买入滑点和卖出滑点都不能为负数。")

    if params.use_ma_filter:
        if params.fast_ma_period < 1 or params.slow_ma_period < 1:
            errors.append("均线周期必须大于等于 1。")
        if params.fast_ma_period > params.slow_ma_period:
            warnings.append(
                "快线周期大于慢线周期是允许的，但请确认这符合您的研究习惯。"
            )

    for canonical, label in (
        ("date", "日期列名"),
        ("stock_code", "股票代码列名"),
        ("open", "开盘价列名"),
        ("high", "最高价列名"),
        ("low", "最低价列名"),
        ("close", "收盘价列名"),
        ("volume", "成交量列名"),
    ):
        if (
            canonical in params.column_overrides
            and not params.column_overrides[canonical].strip()
        ):
            errors.append(f"{label}不能为空。")

    if params.stop_loss_pct >= 100:
        warnings.append("止损比例大于等于 100%，这通常不符合常见交易设置。")

    if params.take_profit_pct >= 100:
        warnings.append("止盈比例大于等于 100%，请确认这是否为预期值。")

    if params.time_stop_target_pct < -100:
        warnings.append("到期最低目标涨幅过低，请确认是否填写正确。")

    if params.start_date > params.end_date:
        errors.append("开始日期不能晚于结束日期。")

    scan_config = params.scan_config
    if scan_config.enabled:
        if not (1 <= len(scan_config.axes) <= 2):
            errors.append("参数扫描当前只支持 1 到 2 个扫描维度。")
        field_names = [axis.field_name for axis in scan_config.axes]
        if len(field_names) != len(set(field_names)):
            errors.append("参数扫描字段不能重复。")
        for axis in scan_config.axes:
            if axis.field_name not in SCAN_FIELD_CASTERS:
                errors.append(f"参数扫描字段不支持：{axis.field_name}")
                continue
            if not axis.values:
                errors.append(f"参数扫描字段 {axis.field_name} 必须提供至少一个值。")
            eligible_fields = FACTOR_SCAN_ELIGIBLE_FIELDS.get(params.entry_factor)
            if eligible_fields is not None and axis.field_name not in eligible_fields:
                errors.append(
                    f"entry_factor={params.entry_factor} 不支持扫描字段：{axis.field_name}"
                )
        if scan_config.metric not in SCAN_METRICS:
            errors.append("参数扫描排序指标不合法。")
        if scan_config.max_combinations < 1:
            errors.append("参数扫描最大组合数必须大于等于 1。")
        if scan_config.combination_count > scan_config.max_combinations:
            errors.append("参数扫描组合数超出上限，请减少扫描值数量。")

    return errors, warnings
