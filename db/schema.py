import os
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Date, BigInteger, Text, Index, UniqueConstraint
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from db.connection import get_engine

load_dotenv()

Base = declarative_base()


class Universe(Base):
    __tablename__ = "universe"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    ticker      = Column(String(16), nullable=False, unique=True)
    name        = Column(Text)
    sector      = Column(String(64))
    industry    = Column(String(128))
    market_cap  = Column(Float)
    is_active   = Column(Integer, default=1)
    added_date  = Column(Date)


class OHLCVDaily(Base):
    __tablename__ = "ohlcv_daily"
    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker      = Column(String(16), nullable=False)
    date        = Column(Date, nullable=False)
    open        = Column(Float)
    high        = Column(Float)
    low         = Column(Float)
    close       = Column(Float)
    adj_close   = Column(Float)
    volume      = Column(BigInteger)
    return_1d   = Column(Float)
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_ohlcv_ticker_date"),
        Index("ix_ohlcv_date", "date"),
        Index("ix_ohlcv_ticker", "ticker"),
    )


class MacroDaily(Base):
    __tablename__ = "macro_daily"
    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    date            = Column(Date, nullable=False, unique=True)
    vix             = Column(Float)
    fed_funds_rate  = Column(Float)
    treasury_10y    = Column(Float)
    treasury_2y     = Column(Float)
    yield_spread    = Column(Float)
    usd_index       = Column(Float)
    sp500_return    = Column(Float)
    __table_args__ = (
        Index("ix_macro_date", "date"),
    )


class MarketPhysicsDaily(Base):
    __tablename__ = "market_physics_daily"
    id                  = Column(BigInteger, primary_key=True, autoincrement=True)
    date                = Column(Date, nullable=False, unique=True)
    temperature         = Column(Float)
    entropy             = Column(Float)
    mean_acceleration   = Column(Float)
    order_parameter     = Column(Float)
    lambda_1            = Column(Float)
    lambda_2            = Column(Float)
    lambda_ratio        = Column(Float)
    d_order_parameter   = Column(Float)
    pe_market           = Column(Float)
    __table_args__ = (
        Index("ix_market_physics_date", "date"),
    )


class SectorPhysicsDaily(Base):
    __tablename__ = "sector_physics_daily"
    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    date        = Column(Date, nullable=False)
    sector      = Column(String(64), nullable=False)
    temperature = Column(Float)
    entropy     = Column(Float)
    pe_sector   = Column(Float)
    mean_accel  = Column(Float)
    stock_count = Column(Integer)
    __table_args__ = (
        UniqueConstraint("date", "sector", name="uq_sector_physics_date_sector"),
        Index("ix_sector_physics_date", "date"),
    )


class StockPhysicsDaily(Base):
    __tablename__ = "stock_physics_daily"
    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker           = Column(String(16), nullable=False)
    date             = Column(Date, nullable=False)
    potential_energy = Column(Float)
    velocity         = Column(Float)
    acceleration     = Column(Float)
    jerk             = Column(Float)
    vol_10d          = Column(Float)
    vol_60d          = Column(Float)
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_stock_physics_ticker_date"),
        Index("ix_stock_physics_date", "date"),
        Index("ix_stock_physics_ticker", "ticker"),
    )


class GNNEmbeddingsDaily(Base):
    __tablename__ = "gnn_embeddings_daily"
    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker      = Column(String(16), nullable=False)
    date        = Column(Date, nullable=False)
    embedding   = Column(JSONB)
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_gnn_ticker_date"),
        Index("ix_gnn_date", "date"),
    )


class LatentStateDaily(Base):
    __tablename__ = "latent_state_daily"
    id       = Column(BigInteger, primary_key=True, autoincrement=True)
    date     = Column(Date, nullable=False, unique=True)
    z_vector = Column(JSONB)
    z_mu     = Column(JSONB)
    z_logvar = Column(JSONB)
    __table_args__ = (
        Index("ix_latent_date", "date"),
    )


class RegimeDaily(Base):
    __tablename__ = "regime_daily"
    id             = Column(BigInteger, primary_key=True, autoincrement=True)
    date           = Column(Date, nullable=False, unique=True)
    regime_label   = Column(Integer)
    regime_probs   = Column(JSONB)
    regime_name    = Column(String(32))
    __table_args__ = (
        Index("ix_regime_date", "date"),
    )


class PortfolioDaily(Base):
    __tablename__ = "portfolio_daily"
    id      = Column(BigInteger, primary_key=True, autoincrement=True)
    date    = Column(Date, nullable=False)
    ticker  = Column(String(16), nullable=False)
    weight  = Column(Float)
    regime  = Column(Integer)
    __table_args__ = (
        UniqueConstraint("date", "ticker", name="uq_portfolio_date_ticker"),
        Index("ix_portfolio_date", "date"),
    )


class BacktestResults(Base):
    __tablename__ = "backtest_results"
    id                 = Column(Integer, primary_key=True, autoincrement=True)
    run_id             = Column(String(64))
    start_date         = Column(Date)
    end_date           = Column(Date)
    total_return       = Column(Float)
    annualized_return  = Column(Float)
    sharpe_ratio       = Column(Float)
    max_drawdown       = Column(Float)
    calmar_ratio       = Column(Float)
    avg_turnover       = Column(Float)
    notes              = Column(Text)


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("All tables created successfully.")


if __name__ == "__main__":
    init_db()
