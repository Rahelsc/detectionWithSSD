package com.rachel.detectionwithssd;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.rachel.detectionwithssd.ml.MobilenetV110224Quant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class MainActivity extends AppCompatActivity {

    private static final int RESULT_PIC = 100;
    private ImageView chosenImage;
    private ArrayList<String> arr;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        arr= new ArrayList<>();
        // reading the labels from the file
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt"),"UTF-8"));
            String mline;
            while ((mline=reader.readLine())!=null){
                    arr.add(mline);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void selectPic(View view) {
        Intent  photoPickerIntent = new Intent(Intent.ACTION_PICK); // create an intent that allow selecting a file
        photoPickerIntent.setType("image/*"); // limiting the file for images only
        startActivityForResult(photoPickerIntent, RESULT_PIC); // action on chosen image
    }


    // ctrl + o --> will show all available built in methods to override

    // what happens after the picture is selected:
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        chosenImage = findViewById(R.id.chosenImage);
        if (resultCode == RESULT_OK){
            final Uri imageUri = data.getData(); // uri of file uploaded
            try {
                final InputStream imageStream = getContentResolver().openInputStream(imageUri); // open a channel to the particular image uploaded
                final Bitmap selectedImage = BitmapFactory.decodeStream(imageStream) ; // getting the actual image
                chosenImage.setImageBitmap(selectedImage); // putting image on chosen view
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
        else
            Toast.makeText(this, "you need to pick an image", Toast.LENGTH_LONG).show();
    }

    public void detect(View view) {
        TextView detectionView = findViewById(R.id.tag);
        try {
            Bitmap bm = ((BitmapDrawable) chosenImage.getDrawable()).getBitmap(); // getting the pic from view
            Bitmap resizedPic = Bitmap.createScaledBitmap(bm,224, 224, true); // downsizing the picture.
            // the size to which we downsize must be the size that appears in the tensorflow model
            TensorImage selectImage = TensorImage.fromBitmap(resizedPic);

            MobilenetV110224Quant model = MobilenetV110224Quant.newInstance(this); // "this" is the context

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8); // here you can see the size of pic the model expects

            ByteBuffer byteBuffer = selectImage.getBuffer(); // select the pic and transfer to bytebuffer (the ssd matrix)

            inputFeature0.loadBuffer(byteBuffer); // instead of bitmap representation, must be byte buffer

            // Runs model inference and gets result.
            MobilenetV110224Quant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer(); // the actual result
            float[] orderedArr = find3Max(outputFeature0.getFloatArray());
            detectionView.setText("");
            // append to textView the 3 highest results
            for (int i = 0; i<3; i++){
                detectionView.append(arr.get(i) + " " + orderedArr[i] + "\n");
            }
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    // return max result
//    public int findMax (float[] arr ){
//        int index = 0;
//        float max = 0;
//        Log.d("hedva", Arrays.toString(arr));
//        for (int i=0; i< arr.length; i++){
//            if (arr[i]> max){
//                index = i;
//                max = arr[i];
//            }
//        }
//        return index;
//    }

    public float[] find3Max (float[] probability ){
        float temp;
        String tempDetectedObjectName;
        for (int i=0; i< probability.length-1; i++){
            for (int j=i+1; j<probability.length; j++)
            if (probability[i]< probability[j]){
                // order the numbers array
                temp = probability[i];
                probability[i] = probability[j];
                probability[j] = temp;
                // order the object array accordingly
                tempDetectedObjectName = arr.get(i);
                arr.set(i, arr.get(j));
                arr.set(j, tempDetectedObjectName);
            }
        }
        return probability;
    }
}