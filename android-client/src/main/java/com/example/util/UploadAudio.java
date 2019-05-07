package com.example.util;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public final class UploadAudio {

    private final OkHttpClient client = new OkHttpClient();
    private static final MediaType MEDIA_TYPE_WAV = MediaType.parse("audio/wave");

    public void upload(String path, String username, String route) {
        File file = new File(path);

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("username", username)
                .addFormDataPart("audio_file", "audio_file.wav",
                        RequestBody.create(MEDIA_TYPE_WAV, new File(path)))
                .build();

        Request request = new Request.Builder()
                .url("http://13.80.124.98:5000/" + route)
                .post(requestBody)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                e.printStackTrace();
            }

            @Override
            public void onResponse(Call call, final Response response) throws IOException {
                if (!response.isSuccessful()) {
                    throw new IOException("Unexpected code " + response);
                }
                String responseList = response.body().string();
                Log.i("OKHTTP3", responseList);
            }
        });
    }

}
