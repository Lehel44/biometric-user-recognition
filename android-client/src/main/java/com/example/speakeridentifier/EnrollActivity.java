package com.example.speakeridentifier;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.example.util.UploadAudio;
import com.example.util.UploadFileAsync;
import com.example.util.WavRecorder;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import okhttp3.MultipartBody;

public class EnrollActivity extends AppCompatActivity {

    private WavRecorder wavRecorder;
    private Button record;
    private EditText name;
    private TextView info;
    private static final String OUTPUT_FILE = Environment.getExternalStorageDirectory().getAbsolutePath() + "/recording.wav";
    private static final String ROUTE = "enroll";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_enroll);

        record = (Button) findViewById(R.id.record);
        name = (EditText) findViewById(R.id.name);
        info = (TextView) findViewById(R.id.info);

        wavRecorder = new WavRecorder(OUTPUT_FILE);
    }

    public void details(View view) {

    }

    public void record(View view) {
        record.setEnabled(false);
        final Handler handler = new Handler();

        Runnable textUpdate = new Runnable() {
            private int sec = 3;
            public void run() {
                info.setText("Start talking in " + sec + "...");
                if (sec == 0) {
                    info.setText("Recording...");
                }
                if (sec == -1) {
                    info.setVisibility(View.INVISIBLE);
                }
                sec--;
            }
        };
        handler.postDelayed(textUpdate, 2000);
        handler.postDelayed(textUpdate, 4000);
        handler.postDelayed(textUpdate, 6000);
        handler.postDelayed(textUpdate, 8000);
        handler.postDelayed(textUpdate, 12000);

        Runnable startAudioRecorder = new Runnable() {
            @Override
            public void run() {
                wavRecorder.startRecording();
                Toast.makeText(getApplicationContext(), "Recording started.", Toast.LENGTH_LONG).show();

            }
        };

        handler.postDelayed(startAudioRecorder, 8000);

        Runnable stopAudioRecorder = new Runnable() {
            @Override
            public void run() {
                wavRecorder.stopRecording();
                Toast.makeText(getApplicationContext(), "Recording stopped.", Toast.LENGTH_LONG).show();
                // Start upload wav.
                //new UploadFileAsync(name.getText().toString(), ROUTE).execute("");
                new UploadAudio().upload(OUTPUT_FILE, name.getText().toString(), "enroll");
            }
        };

        handler.postDelayed(stopAudioRecorder, 12000);


    }


}
