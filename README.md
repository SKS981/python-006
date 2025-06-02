import re
import time
import threading
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
from typing import Optional, Dict

required = [
    ('yfinance', 'yfinance'),
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('sklearn', 'scikit-learn'),
    ('statsmodels', 'statsmodels'),
    # 可选：akshare
]
missing = []
for pkg, pip_name in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append((pkg, pip_name))
if missing:
    msg = "缺少依赖包：\n"
    msg += "\n".join([f"{pkg}（pip install {pip_name}）" for pkg, pip_name in missing])
    tk.Tk().withdraw()
    messagebox.showerror("依赖缺失", msg)
    sys.exit(1)

import yfinance as yf
import pandas as pd
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

PROXY_POOL = [
    {"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"},
]

class StockDataManager:
    def __init__(self):
        self.data = None
        self.ticker = None
        self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.proxy_pool = PROXY_POOL.copy()

    def format_stock_code(self, raw_code: str) -> str:
        clean_code = re.sub(r'[^a-zA-Z0-9.]', '', raw_code).upper()
        if re.match(r'^\d{6}$', clean_code):
            if clean_code.startswith('6'):
                return clean_code + ".SS"
            else:
                return clean_code + ".SZ"
        return clean_code

    def _validate_proxy(self, proxy: Dict) -> bool:
        try:
            test_url = "https://finance.yahoo.com"
            response = self.session.get(test_url, proxies=proxy, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def fetch_stock_data(self, code: str, start: str, end: str, proxy: Optional[Dict] = None) -> bool:
        self.ticker = self.format_stock_code(code)
        cache_key = f"{self.ticker}|{start}|{end}|{proxy}"
        if cache_key in self.cache:
            self.data = self.cache[cache_key]
            return True
        raw_code = re.sub(r'\D', '', code)
        # 优先尝试 akshare
        if len(raw_code) == 6:
            if self._try_akshare(raw_code, start, end):
                self.cache[cache_key] = self.data
                return True
        proxy_list = []
        if proxy:
            proxy_list.append(proxy)
        proxy_list.extend(self.proxy_pool)
        proxy_list.append(None)
        for current_proxy in proxy_list:
            if current_proxy:
                self.session.proxies.update(current_proxy)
                if not self._validate_proxy(current_proxy):
                    continue
            else:
                self.session.proxies.clear()
            for attempt in range(3):
                try:
                    if attempt > 0:
                        time.sleep(3 * attempt)
                    data = yf.download(
                        self.ticker,
                        start=start,
                        end=end,
                        progress=False,
                        session=self.session,
                        timeout=20
                    )
                    if not data.empty:
                        self.data = data
                        self.cache[cache_key] = data
                        return True
                except Exception:
                    if attempt == 2:
                        break
        return False

    def _try_akshare(self, raw_code, start, end):
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(
                symbol=raw_code,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust=""
            )
            if not df.empty:
                df = df.rename(columns={
                    '日期': 'Date', '开盘': 'Open', '最高': 'High',
                    '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'
                }).set_index('Date')
                df.index = pd.to_datetime(df.index)
                self.data = df
                return True
        except Exception:
            pass
        return False

    def predict_future_signal(self, df, future_days=30):
        """
        增强版：用线性回归和ARIMA融合预测未来收盘价，计算未来MA5/MA20金叉死叉，输出未来N天信号
        """
        import warnings
        warnings.filterwarnings("ignore")
        df = df.copy().dropna()
        last_N = 60
        recent_df = df[-last_N:]
        if len(recent_df) < 30:
            return "历史数据不足，无法预测未来买卖信号。"
        y = recent_df['Close'].values

        # 1. 线性回归预测
        X = np.arange(len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(X, y)
        future_X = np.arange(len(y), len(y) + future_days).reshape(-1, 1)
        pred_lr = lr.predict(future_X)

        # 2. ARIMA预测
        pred_arima = []
        try:
            arima_model = ARIMA(y, order=(3,1,0))
            arima_fit = arima_model.fit()
            pred_arima = arima_fit.forecast(steps=future_days)
        except Exception:
            pred_arima = pred_lr

        # 3. 融合
        if isinstance(pred_arima, np.ndarray) and pred_arima.shape == pred_lr.shape:
            pred = (pred_lr + pred_arima) / 2
        else:
            pred = pred_lr

        # 4. MA均线
        full_close = np.concatenate([y, pred])
        full_close_ser = pd.Series(full_close)
        full_ma5 = full_close_ser.rolling(window=5).mean().values
        full_ma20 = full_close_ser.rolling(window=20).mean().values

        # 5. 检查未来区间金叉/死叉
        signals = []
        for i in range(len(y), len(full_close)):
            idx = i
            if idx < 20: continue
            prev_ma5, prev_ma20 = full_ma5[idx-1], full_ma20[idx-1]
            now_ma5, now_ma20 = full_ma5[idx], full_ma20[idx]
            if np.isnan(prev_ma5) or np.isnan(prev_ma20) or np.isnan(now_ma5) or np.isnan(now_ma20):
                continue
            # 金叉
            if prev_ma5 <= prev_ma20 and now_ma5 > now_ma20:
                signals.append((i-len(y)+1, "买入"))
            # 死叉
            if prev_ma5 >= prev_ma20 and now_ma5 < now_ma20:
                signals.append((i-len(y)+1, "卖出"))
        if not signals:
            return f"预测未来{future_days}天未发现买入或卖出信号，建议观望。"
        msg = []
        for offset, sigtype in signals:
            msg.append(f"第{offset}天后可能出现“{sigtype}”信号")
        return f"预测未来{future_days}天：" + "，".join(msg) + "，请重点关注。"

# ========== GUI界面类 ==========
class StockGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("未来买卖时机预测工具")
        self.master.geometry("1000x750")
        self.master.resizable(True, True)
        self.data_manager = StockDataManager()
        self._create_widgets()
        self._setup_validation()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.master, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.LabelFrame(main_frame, text="数据参数")
        input_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)

        ttk.Label(input_frame, text="股票代码:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.code_entry = ttk.Entry(input_frame, width=25)
        self.code_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.code_entry.insert(0, "600519")

        ttk.Label(input_frame, text="开始日期:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.start_entry = ttk.Entry(input_frame)
        self.start_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self.start_entry.insert(0, (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"))

        ttk.Label(input_frame, text="结束日期:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.end_entry = ttk.Entry(input_frame)
        self.end_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self.end_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))

        ttk.Label(input_frame, text="预测未来天数:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.days_entry = ttk.Entry(input_frame, width=10)
        self.days_entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        self.days_entry.insert(0, "30")

        proxy_frame = ttk.LabelFrame(main_frame, text="代理服务器（可选）")
        proxy_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=5, pady=5)

        ttk.Label(proxy_frame, text="地址:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.proxy_addr_entry = ttk.Entry(proxy_frame, width=25)
        self.proxy_addr_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.proxy_addr_entry.insert(0, "http://127.0.0.1")

        ttk.Label(proxy_frame, text="端口:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.proxy_port_entry = ttk.Entry(proxy_frame, width=10)
        self.proxy_port_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.proxy_port_entry.insert(0, "10809")

        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=2, column=0, sticky=tk.NSEW, padx=5, pady=10)

        ttk.Button(btn_frame, text="获取数据与未来预测", command=self._fetch_data_thread, width=22).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="清空缓存", command=self._clear_cache, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="查看数据", command=self._view_data, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="退出", command=self.master.quit, width=15).pack(side=tk.RIGHT, padx=5)

        log_frame = ttk.LabelFrame(main_frame, text="操作日志")
        log_frame.grid(row=0, column=1, rowspan=3, sticky=tk.NSEW, padx=5, pady=5)

        self.log_text = tk.Text(log_frame, height=28, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_scroll = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=self.log_scroll.set, state=tk.DISABLED)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

    def _setup_validation(self):
        date_re = r'^\d{4}-\d{2}-\d{2}$'
        validator = self.master.register(lambda s: re.fullmatch(date_re, s) is not None)
        self.start_entry.config(validate="focusout", validatecommand=(validator, '%P'), invalidcommand=self._show_date_error)
        self.end_entry.config(validate="focusout", validatecommand=(validator, '%P'), invalidcommand=self._show_date_error)

    def _show_date_error(self):
        messagebox.showerror("格式错误", "请输入正确的日期格式（YYYY-MM-DD）")

    def _log_message(self, msg: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_cache(self):
        self.data_manager.cache.clear()
        self._log_message("本地缓存已清空")
        messagebox.showinfo("提示", "已清空本地数据缓存")

    def _view_data(self):
        if self.data_manager.data is None or self.data_manager.data.empty:
            messagebox.showinfo("提示", "暂无数据，请先获取数据")
            return
        try:
            data_window = tk.Toplevel(self.master)
            data_window.title(f"数据预览 - {self.data_manager.ticker}")
            data_window.geometry("1000x500")
            columns = ['Date'] + list(self.data_manager.data.columns)
            tree = ttk.Treeview(data_window, columns=columns, show="headings")
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100, anchor=tk.CENTER)
            for i, row in self.data_manager.data.iterrows():
                tree.insert("", tk.END, values=[str(i)] + list(row))
            scroll_y = ttk.Scrollbar(data_window, orient=tk.VERTICAL, command=tree.yview)
            scroll_x = ttk.Scrollbar(data_window, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscroll=scroll_y.set, xscroll=scroll_x.set)
            scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            tree.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("错误", f"显示数据失败: {str(e)}")

    def _fetch_data_thread(self):
        t = threading.Thread(target=self._fetch_data)
        t.daemon = True
        t.start()

    def _fetch_data(self):
        start_date = self.start_entry.get().strip()
        end_date = self.end_entry.get().strip()
        if not re.fullmatch(r'^\d{4}-\d{2}-\d{2}$', start_date) or \
                not re.fullmatch(r'^\d{4}-\d{2}-\d{2}$', end_date):
            messagebox.showerror("输入错误", "日期格式不正确")
            return
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if start_dt >= end_dt:
                messagebox.showerror("输入错误", "开始日期必须早于结束日期")
                return
        except ValueError as e:
            messagebox.showerror("输入错误", f"日期解析错误: {str(e)}")
            return
        # 预测天数
        try:
            future_days = int(self.days_entry.get().strip())
            if not (1 <= future_days <= 90):
                raise ValueError()
        except Exception:
            messagebox.showerror("输入错误", "预测未来天数需为1-90之间的整数")
            return

        self._log_message("开始获取数据...")
        code = self.code_entry.get().strip()
        proxy = None
        proxy_addr = self.proxy_addr_entry.get().strip()
        proxy_port = self.proxy_port_entry.get().strip()
        if proxy_addr and proxy_port:
            proxy = {
                "http": f"{proxy_addr}:{proxy_port}",
                "https": f"{proxy_addr}:{proxy_port}"
            }
        result = self.data_manager.fetch_stock_data(code, start_date, end_date, proxy)
        if result and self.data_manager.data is not None and not self.data_manager.data.empty:
            self._log_message(f"成功获取 {len(self.data_manager.data)} 条数据")
            # 未来买卖信号预测
            prediction = self.data_manager.predict_future_signal(self.data_manager.data, future_days=future_days)
            self._log_message(prediction)
            messagebox.showinfo("未来买卖时机预测", prediction)
            try:
                csv_file = f"{code}_{start_date}_{end_date}.csv"
                self.data_manager.data.to_csv(csv_file)
                self._log_message(f"数据已保存到 {csv_file}")
            except Exception as e:
                self._log_message(f"保存CSV失败: {str(e)}")
        else:
            self._log_message("数据获取失败")
            messagebox.showerror("错误", f"无法获取数据\n代码: {code}\n请检查代理、网络或股票代码是否正确")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockGUI(root)
    root.mainloop()
