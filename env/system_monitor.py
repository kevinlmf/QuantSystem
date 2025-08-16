"""
Comprehensive System Monitoring and Logging
Real-time monitoring, alerting, and performance tracking
"""
import logging
import json
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import warnings
warnings.filterwarnings('ignore')

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"

class MetricType(Enum):
    PERFORMANCE = "performance"
    RISK = "risk"
    SYSTEM = "system"
    EXECUTION = "execution"
    DATA_QUALITY = "data_quality"

@dataclass
class Alert:
    """System alert"""
    timestamp: datetime
    level: AlertLevel
    metric_type: MetricType
    message: str
    details: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    uptime_hours: float

@dataclass
class TradingMetrics:
    """Trading system metrics"""
    portfolio_value: float
    daily_return: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    active_positions: int
    pending_orders: int
    execution_latency_ms: float
    fill_rate: float

class SystemMonitor:
    """
    Comprehensive system monitoring with real-time alerts and logging
    
    Features:
    - Real-time system metrics collection
    - Performance monitoring and alerting
    - Risk monitoring with breach detection
    - Execution quality monitoring
    - Data quality monitoring
    - Custom alert rules and notifications
    - Historical metrics storage
    - Dashboard data generation
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 alert_thresholds: Dict[str, float] = None,
                 email_config: Dict[str, str] = None,
                 monitoring_interval: int = 60,
                 max_log_files: int = 30):
        """
        Initialize system monitor
        
        Args:
            log_dir: Directory for log files
            alert_thresholds: Thresholds for various alerts
            email_config: Email configuration for alerts
            monitoring_interval: Monitoring interval in seconds
            max_log_files: Maximum number of log files to keep
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'max_drawdown': 0.05,
            'var_95': 0.02,
            'execution_latency_ms': 1000.0,
            'fill_rate_min': 0.95,
            'sharpe_ratio_min': 0.5
        }
        
        self.email_config = email_config
        self.monitoring_interval = monitoring_interval
        self.max_log_files = max_log_files
        
        # Initialize logging
        self._setup_logging()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.start_time = datetime.now()
        
        # Metrics storage
        self.system_metrics_history = []
        self.trading_metrics_history = []
        self.alerts = []
        self.alert_rules = []
        
        # Performance tracking
        self.last_metrics_time = datetime.now()
        self.metrics_collection_times = []
        
        # Custom callbacks
        self.metric_callbacks = {}
        
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        # Main system logger
        self.logger = logging.getLogger('trading_system')
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # System log file handler
        system_log_file = self.log_dir / f"system_{datetime.now().strftime('%Y%m%d')}.log"
        system_handler = logging.FileHandler(system_log_file)
        system_handler.setLevel(logging.DEBUG)
        system_handler.setFormatter(detailed_formatter)
        
        # Trading log file handler
        trading_log_file = self.log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        self.trading_logger = logging.getLogger('trading_system.trading')
        trading_handler = logging.FileHandler(trading_log_file)
        trading_handler.setLevel(logging.INFO)
        trading_handler.setFormatter(detailed_formatter)
        
        # Performance log file handler
        performance_log_file = self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
        self.performance_logger = logging.getLogger('trading_system.performance')
        performance_handler = logging.FileHandler(performance_log_file)
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(simple_formatter)
        
        # Alert log file handler
        alert_log_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.log"
        self.alert_logger = logging.getLogger('trading_system.alerts')
        alert_handler = logging.FileHandler(alert_log_file)
        alert_handler.setLevel(logging.WARNING)
        alert_handler.setFormatter(detailed_formatter)
        
        # Console handler for critical alerts
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.CRITICAL)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers to loggers
        self.logger.addHandler(system_handler)
        self.logger.addHandler(console_handler)
        self.trading_logger.addHandler(trading_handler)
        self.performance_logger.addHandler(performance_handler)
        self.alert_logger.addHandler(alert_handler)
        self.alert_logger.addHandler(console_handler)
        
        self.logger.info("System monitoring initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append({
                    'timestamp': datetime.now(),
                    **asdict(system_metrics)
                })
                
                # Check system alerts
                self._check_system_alerts(system_metrics)
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Track collection performance
                collection_time = time.time() - start_time
                self.metrics_collection_times.append(collection_time)
                if len(self.metrics_collection_times) > 100:
                    self.metrics_collection_times.pop(0)
                
                # Sleep until next collection
                time.sleep(max(0, self.monitoring_interval - collection_time))
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # Process count
        process_count = len(psutil.pids())
        
        # Uptime
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            process_count=process_count,
            uptime_hours=uptime_hours
        )
    
    def log_trading_metrics(self, metrics: TradingMetrics):
        """Log trading system metrics"""
        # Store metrics
        self.trading_metrics_history.append({
            'timestamp': datetime.now(),
            **asdict(metrics)
        })
        
        # Log to performance logger
        self.performance_logger.info(
            f"Portfolio: ${metrics.portfolio_value:,.2f} | "
            f"Daily Return: {metrics.daily_return:.2%} | "
            f"Total Return: {metrics.total_return:.2%} | "
            f"Sharpe: {metrics.sharpe_ratio:.2f} | "
            f"Max DD: {metrics.max_drawdown:.2%} | "
            f"Positions: {metrics.active_positions}"
        )
        
        # Check trading alerts
        self._check_trading_alerts(metrics)
        
        # Execute custom callbacks
        for callback_name, callback_func in self.metric_callbacks.items():
            try:
                callback_func(metrics)
            except Exception as e:
                self.logger.error(f"Error in metric callback {callback_name}: {e}")
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system-related alerts"""
        alerts_triggered = []
        
        # CPU usage alert
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts_triggered.append(Alert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                metric_type=MetricType.SYSTEM,
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                details={'cpu_usage': metrics.cpu_usage, 'threshold': self.alert_thresholds['cpu_usage']}
            ))
        
        # Memory usage alert
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts_triggered.append(Alert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                metric_type=MetricType.SYSTEM,
                message=f"High memory usage: {metrics.memory_usage:.1f}%",
                details={'memory_usage': metrics.memory_usage, 'threshold': self.alert_thresholds['memory_usage']}
            ))
        
        # Disk usage alert
        if metrics.disk_usage > self.alert_thresholds['disk_usage']:
            alerts_triggered.append(Alert(
                timestamp=datetime.now(),
                level=AlertLevel.CRITICAL,
                metric_type=MetricType.SYSTEM,
                message=f"High disk usage: {metrics.disk_usage:.1f}%",
                details={'disk_usage': metrics.disk_usage, 'threshold': self.alert_thresholds['disk_usage']}
            ))
        
        # Process alerts
        for alert in alerts_triggered:
            self._process_alert(alert)
    
    def _check_trading_alerts(self, metrics: TradingMetrics):
        """Check for trading-related alerts"""
        alerts_triggered = []
        
        # Max drawdown alert
        if abs(metrics.max_drawdown) > self.alert_thresholds['max_drawdown']:
            alerts_triggered.append(Alert(
                timestamp=datetime.now(),
                level=AlertLevel.CRITICAL,
                metric_type=MetricType.RISK,
                message=f"Maximum drawdown exceeded: {metrics.max_drawdown:.2%}",
                details={'max_drawdown': metrics.max_drawdown, 'threshold': self.alert_thresholds['max_drawdown']}
            ))
        
        # VaR alert
        if abs(metrics.var_95) > self.alert_thresholds['var_95']:
            alerts_triggered.append(Alert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                metric_type=MetricType.RISK,
                message=f"VaR 95% exceeded: {metrics.var_95:.2%}",
                details={'var_95': metrics.var_95, 'threshold': self.alert_thresholds['var_95']}
            ))
        
        # Execution latency alert
        if metrics.execution_latency_ms > self.alert_thresholds['execution_latency_ms']:
            alerts_triggered.append(Alert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                metric_type=MetricType.EXECUTION,
                message=f"High execution latency: {metrics.execution_latency_ms:.1f}ms",
                details={'latency_ms': metrics.execution_latency_ms, 'threshold': self.alert_thresholds['execution_latency_ms']}
            ))
        
        # Fill rate alert
        if metrics.fill_rate < self.alert_thresholds['fill_rate_min']:
            alerts_triggered.append(Alert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                metric_type=MetricType.EXECUTION,
                message=f"Low fill rate: {metrics.fill_rate:.2%}",
                details={'fill_rate': metrics.fill_rate, 'threshold': self.alert_thresholds['fill_rate_min']}
            ))
        
        # Sharpe ratio alert (if significantly negative)
        if metrics.sharpe_ratio < -1.0:
            alerts_triggered.append(Alert(
                timestamp=datetime.now(),
                level=AlertLevel.CRITICAL,
                metric_type=MetricType.PERFORMANCE,
                message=f"Very poor Sharpe ratio: {metrics.sharpe_ratio:.2f}",
                details={'sharpe_ratio': metrics.sharpe_ratio}
            ))
        
        # Process alerts
        for alert in alerts_triggered:
            self._process_alert(alert)
    
    def _process_alert(self, alert: Alert):
        """Process and handle an alert"""
        # Add to alerts list
        self.alerts.append(alert)
        
        # Log alert
        if alert.level == AlertLevel.CRITICAL:
            self.alert_logger.critical(f"{alert.metric_type.value.upper()}: {alert.message}")
        elif alert.level == AlertLevel.ERROR:
            self.alert_logger.error(f"{alert.metric_type.value.upper()}: {alert.message}")
        elif alert.level == AlertLevel.WARNING:
            self.alert_logger.warning(f"{alert.metric_type.value.upper()}: {alert.message}")
        else:
            self.alert_logger.info(f"{alert.metric_type.value.upper()}: {alert.message}")
        
        # Send email notification if configured
        if self.email_config and alert.level in [AlertLevel.CRITICAL, AlertLevel.ERROR]:
            self._send_email_alert(alert)
        
        # Execute custom alert handlers
        self._execute_alert_handlers(alert)
    
    def _send_email_alert(self, alert: Alert):
        """Send email notification for critical alerts"""
        if not self.email_config:
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = f"Trading System Alert - {alert.level.value.upper()}"
            
            body = f"""
Trading System Alert

Level: {alert.level.value.upper()}
Type: {alert.metric_type.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Message: {alert.message}

Details:
{json.dumps(alert.details, indent=2)}

System Status:
- Uptime: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f} hours
- Active Monitoring: {self.is_monitoring}

Please investigate immediately.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            if self.email_config.get('use_tls', True):
                server.starttls()
            if 'username' in self.email_config:
                server.login(self.email_config['username'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], self.email_config['to_email'], text)
            server.quit()
            
            self.logger.info(f"Alert email sent for: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert email: {e}")
    
    def _execute_alert_handlers(self, alert: Alert):
        """Execute custom alert handlers"""
        # This can be extended with custom alert handling logic
        pass
    
    def add_custom_metric_callback(self, name: str, callback: Callable[[TradingMetrics], None]):
        """Add custom callback for metric processing"""
        self.metric_callbacks[name] = callback
        self.logger.info(f"Added custom metric callback: {name}")
    
    def remove_custom_metric_callback(self, name: str):
        """Remove custom metric callback"""
        if name in self.metric_callbacks:
            del self.metric_callbacks[name]
            self.logger.info(f"Removed custom metric callback: {name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status summary"""
        current_time = datetime.now()
        
        # Get latest metrics
        latest_system = self.system_metrics_history[-1] if self.system_metrics_history else None
        latest_trading = self.trading_metrics_history[-1] if self.trading_metrics_history else None
        
        # Count recent alerts
        recent_alerts = [a for a in self.alerts if a.timestamp > current_time - timedelta(hours=24)]
        alert_counts = {level.value: 0 for level in AlertLevel}
        for alert in recent_alerts:
            alert_counts[alert.level.value] += 1
        
        # Collection performance
        avg_collection_time = np.mean(self.metrics_collection_times) if self.metrics_collection_times else 0
        
        return {
            'timestamp': current_time,
            'uptime_hours': (current_time - self.start_time).total_seconds() / 3600,
            'monitoring_active': self.is_monitoring,
            'system_metrics': latest_system,
            'trading_metrics': latest_trading,
            'alerts_24h': alert_counts,
            'total_alerts': len(self.alerts),
            'avg_collection_time_ms': avg_collection_time * 1000,
            'data_points_collected': {
                'system': len(self.system_metrics_history),
                'trading': len(self.trading_metrics_history)
            }
        }
    
    def get_performance_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get data for performance dashboard"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent data
        recent_system = [m for m in self.system_metrics_history if m['timestamp'] > cutoff_time]
        recent_trading = [m for m in self.trading_metrics_history if m['timestamp'] > cutoff_time]
        
        if not recent_system and not recent_trading:
            return {}
        
        # Prepare dashboard data
        dashboard_data = {
            'system_metrics_timeseries': recent_system,
            'trading_metrics_timeseries': recent_trading,
            'current_status': self.get_system_status(),
            'alert_summary': self._get_alert_summary(hours),
        }
        
        if recent_trading:
            # Trading performance summary
            trading_df = pd.DataFrame(recent_trading)
            dashboard_data['performance_summary'] = {
                'current_portfolio_value': trading_df['portfolio_value'].iloc[-1],
                'period_return': (trading_df['portfolio_value'].iloc[-1] / trading_df['portfolio_value'].iloc[0] - 1) if len(trading_df) > 1 else 0,
                'avg_daily_return': trading_df['daily_return'].mean(),
                'volatility': trading_df['daily_return'].std(),
                'max_drawdown': trading_df['max_drawdown'].min(),
                'avg_positions': trading_df['active_positions'].mean(),
            }
        
        return dashboard_data
    
    def _get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for dashboard"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        # Group by type and level
        by_type = {}
        by_level = {level.value: 0 for level in AlertLevel}
        
        for alert in recent_alerts:
            # By type
            type_key = alert.metric_type.value
            if type_key not in by_type:
                by_type[type_key] = 0
            by_type[type_key] += 1
            
            # By level
            by_level[alert.level.value] += 1
        
        return {
            'total_alerts': len(recent_alerts),
            'by_type': by_type,
            'by_level': by_level,
            'recent_alerts': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'level': a.level.value,
                    'type': a.metric_type.value,
                    'message': a.message
                } for a in recent_alerts[-10:]  # Last 10 alerts
            ]
        }
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of data
        
        # Clean system metrics
        self.system_metrics_history = [
            m for m in self.system_metrics_history 
            if m['timestamp'] > cutoff_time
        ]
        
        # Clean trading metrics
        self.trading_metrics_history = [
            m for m in self.trading_metrics_history 
            if m['timestamp'] > cutoff_time
        ]
        
        # Clean alerts (keep 30 days)
        alert_cutoff = datetime.now() - timedelta(days=30)
        self.alerts = [a for a in self.alerts if a.timestamp > alert_cutoff]
        
        # Clean old log files
        self._cleanup_old_log_files()
    
    def _cleanup_old_log_files(self):
        """Clean up old log files"""
        try:
            log_files = list(self.log_dir.glob("*.log"))
            if len(log_files) > self.max_log_files:
                # Sort by modification time and remove oldest
                log_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in log_files[:-self.max_log_files]:
                    old_file.unlink()
                    self.logger.info(f"Deleted old log file: {old_file}")
        except Exception as e:
            self.logger.error(f"Error cleaning up log files: {e}")
    
    def export_metrics_to_csv(self, filepath: str, hours: int = 24):
        """Export recent metrics to CSV"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Export system metrics
        system_data = [m for m in self.system_metrics_history if m['timestamp'] > cutoff_time]
        if system_data:
            system_df = pd.DataFrame(system_data)
            system_df.to_csv(f"{filepath}_system_metrics.csv", index=False)
        
        # Export trading metrics
        trading_data = [m for m in self.trading_metrics_history if m['timestamp'] > cutoff_time]
        if trading_data:
            trading_df = pd.DataFrame(trading_data)
            trading_df.to_csv(f"{filepath}_trading_metrics.csv", index=False)
        
        self.logger.info(f"Metrics exported to {filepath}")

# Global monitor instance
system_monitor = SystemMonitor()