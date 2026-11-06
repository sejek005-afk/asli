package com.asli.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.SharedPreferences;
import android.graphics.Color;
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
import android.widget.Button;
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
    private WebView web;
    private SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences(PREFS, MODE_PRIVATE);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#0f1115"));

        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(16, 8, 16, 8);
        TextView title = new TextView(this);
        title.setText("ASLI");
        title.setTextColor(Color.WHITE);
        title.setTextSize(20);
        bar.addView(title, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        bar.addView(button("داشبورد", v -> openPath("/")));
        bar.addView(button("گزارش", v -> openPath("/report")));
        bar.addView(button("تنظیمات", v -> askSettings()));
        root.addView(bar);

        web = new WebView(this);
        WebSettings ws = web.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true);
        ws.setCacheMode(WebSettings.LOAD_NO_CACHE);
        web.setBackgroundColor(Color.parseColor("#0f1115"));
        web.setWebChromeClient(new WebChromeClient());
        web.setWebViewClient(new WebViewClient() {
            @Override
            public void onReceivedError(WebView view, WebResourceRequest req, WebResourceError err) {
                if (req.isForMainFrame()) {
                    view.loadDataWithBaseURL(null,
                            "<html dir='rtl'><body style='background:#0f1115;color:#eee;font-family:sans-serif;padding:24px'>"
                                    + "<h3>اتصال به سرور برقرار نشد</h3><p>" + err.getDescription() + "</p>"
                                    + "<p>آدرس: " + serverUrl() + "</p><p>بررسی کنید ربات روی سرور روشن است و پورت 8787 در فایروال باز است، "
                                    + "یا از «تنظیمات» آدرس و توکن را درست کنید.</p></body></html>",
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

    private Button button(String text, View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setOnClickListener(l);
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
