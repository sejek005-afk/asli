package com.asli.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

/**
 * ASLI mobile app: shows the bot's live dashboard (account, open positions, every decision of the
 * opposing agent with its arguments, economic calendar) and the backtest report from the server.
 * Controls (pause / resume / close all) are on the dashboard page itself.
 */
public class MainActivity extends Activity {
    private static final String PREFS = "asli";
    private static final String DEFAULT_URL = "http://185.215.244.236:8787";
    private static final String BG = "#0a0d14";
    private WebView web;
    private SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences(PREFS, MODE_PRIVATE);

        int bg = Color.parseColor(BG);
        getWindow().setStatusBarColor(bg);
        getWindow().setNavigationBarColor(bg);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(bg);

        // slim bar: the page has its own header and tabs; here only home, the backtest report and the connection
        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(dp(12), dp(6), dp(12), dp(6));
        TextView title = new TextView(this);
        title.setText("● ASLI");
        title.setTextColor(Color.parseColor("#f5c451"));
        title.setTextSize(15);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        bar.addView(title, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        bar.addView(button("خانه", v -> openPath("/")));
        bar.addView(button("گزارش", v -> openPath("/report")));
        bar.addView(button("اتصال", v -> askSettings()));
        root.addView(bar);

        web = new WebView(this);
        WebSettings ws = web.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true);
        ws.setCacheMode(WebSettings.LOAD_NO_CACHE);
        web.setBackgroundColor(Color.parseColor(BG));
        web.setWebChromeClient(new WebChromeClient());
        web.setWebViewClient(new WebViewClient() {
            @Override
            public void onReceivedError(WebView view, WebResourceRequest req, WebResourceError err) {
                if (req.isForMainFrame()) {
                    view.loadDataWithBaseURL(null,
                            "<html dir='rtl'><body style='background:#0a0d14;color:#e9edf5;font-family:sans-serif;padding:24px'>"
                                    + "<h3>اتصال به سرور برقرار نشد</h3><p>" + err.getDescription() + "</p>"
                                    + "<p>آدرس: " + serverUrl() + "</p><p>بررسی کنید ربات روی سرور روشن است و پورت 8787 در فایروال باز است، "
                                    + "یا از «اتصال» آدرس و توکن را درست کنید.</p></body></html>",
                            "text/html", "utf-8", null);
                }
            }
        });
        root.addView(web, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);

        if (prefs.getString("token", "").isEmpty()) {
            askSettings();
        } else {
            openPath("/");
        }
    }

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }

    private TextView button(String text, View.OnClickListener l) {
        TextView b = new TextView(this);
        b.setText(text);
        b.setTextColor(Color.parseColor("#e9edf5"));
        b.setTextSize(13);
        b.setPadding(dp(12), dp(6), dp(12), dp(6));
        GradientDrawable shape = new GradientDrawable();
        shape.setColor(Color.parseColor("#171f31"));
        shape.setCornerRadius(dp(14));
        shape.setStroke(dp(1), Color.parseColor("#232c40"));
        b.setBackground(shape);
        b.setClickable(true);
        b.setOnClickListener(l);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMarginStart(dp(6));
        b.setLayoutParams(lp);
        return b;
    }

    private String serverUrl() {
        String u = prefs.getString("url", DEFAULT_URL).trim();
        while (u.endsWith("/")) u = u.substring(0, u.length() - 1);
        return u;
    }

    private void openPath(String path) {
        String token = prefs.getString("token", "");
        web.loadUrl(serverUrl() + path + "?token=" + Uri.encode(token));
    }

    private void askSettings() {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(40, 20, 40, 0);
        final EditText url = new EditText(this);
        url.setHint("آدرس سرور");
        url.setText(prefs.getString("url", DEFAULT_URL));
        url.setInputType(InputType.TYPE_TEXT_VARIATION_URI);
        final EditText token = new EditText(this);
        token.setHint("توکن (server.token در config.yaml)");
        token.setText(prefs.getString("token", ""));
        token.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        box.addView(url);
        box.addView(token);
        new AlertDialog.Builder(this)
                .setTitle("اتصال به ربات")
                .setView(box)
                .setPositiveButton("ذخیره", (d, w) -> {
                    prefs.edit().putString("url", url.getText().toString().trim())
                            .putString("token", token.getText().toString().trim()).apply();
                    openPath("/");
                })
                .setNegativeButton("لغو", null)
                .show();
    }

    @Override
    public void onBackPressed() {
        if (web != null && web.canGoBack()) {
            web.goBack();
        } else {
            super.onBackPressed();
        }
    }
}
