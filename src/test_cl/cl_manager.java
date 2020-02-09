/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test_cl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL10.CL_CONTEXT_PLATFORM;
import static org.lwjgl.opencl.CL10.CL_DEVICE_NOT_FOUND;
import static org.lwjgl.opencl.CL10.CL_DEVICE_TYPE_GPU;
import static org.lwjgl.opencl.CL10.clBuildProgram;
import static org.lwjgl.opencl.CL10.clCreateCommandQueue;
import static org.lwjgl.opencl.CL10.clCreateContext;
import static org.lwjgl.opencl.CL10.clCreateKernel;
import static org.lwjgl.opencl.CL10.clEnqueueNDRangeKernel;
import static org.lwjgl.opencl.CL10.clGetDeviceIDs;
import static org.lwjgl.opencl.CL10.clGetPlatformIDs;
import static org.lwjgl.opencl.CL10.clSetKernelArg;
//import static org.lwjgl.opencl.InfoUtil.checkCLError;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memUTF8;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CL10;
import org.lwjgl.opencl.CLCapabilities;
import org.lwjgl.opencl.CLContextCallback;
import org.lwjgl.opencl.CLProgramCallback;
import org.lwjgl.system.MemoryStack;

/**
 *
 * @author dylan
 * create cl script loader.
 */
public class cl_manager {
    private static final String sumProgramSource =
      "kernel void sum(global const float* a, global const float* b, global float* result, int const size) {"
          + "  const int itemId = get_global_id(0);" + "  if(itemId < size) {"
          + "    result[itemId] = a[itemId] + b[itemId];" + "  }" + "}";

    private CLContextCallback clContextCB;
    private long clContext;
    private IntBuffer errcode_ret;
    private long clKernel;
    private long clDevice;
    private CLCapabilities deviceCaps;
    private long clQueue;
    private long sumProgram;
    private long aMemory;
    private long bMemory;
    private long clPlatform;
    private CLCapabilities clPlatformCapabilities;
    private long resultMemory;
    private static final int size = 100;
  
    public cl_manager(){
    }
    
    /** Utility method to convert float array to float buffer
     * @param floats - the float array to convert
     * @return a float buffer containing the input float array
     */
    static FloatBuffer toFloatBuffer(float[] floats) {
        FloatBuffer buf = BufferUtils.createFloatBuffer(floats.length).put(floats);
        buf.rewind();
        return buf;
    }

    /** Utility method to print a float buffer
     * @param buffer - the float buffer to print to System.out
     */
    static void print(FloatBuffer buffer) {
        for (int i = 0; i < buffer.capacity(); i++) {
            System.out.print(buffer.get(i)+" ");
        }
        System.out.println("");
    }
    
     /**
     * Read a resource into a string.
     * @param filePath The resource to read.
     * @return The resource as a string.
     * @throws java.io.IOException
     */
    public static String getResourceAsString(String filePath) throws IOException {
        InputStream is = cl_manager.class.getClassLoader().getResourceAsStream(filePath);
        if (is == null) {
            throw new IOException("Can't find resource: " + filePath);
        }
        BufferedReader br = new BufferedReader(new InputStreamReader(is, "UTF-8"));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) {
            sb.append(line);
        }
        return sb.toString();
    }
    
    static void checkCLError(int errcode) {
        if (errcode != CL_SUCCESS) {
            throw new RuntimeException(String.format("OpenCL error [%d]", errcode));
        }
    }
    
    static void checkCLError(IntBuffer errcode) {
        checkCLError(errcode.get(errcode.position()));
    }
    
    public void run() {
        initializeCL();
        System.out.println("opencl initialized");

        String filePath = "test_cl/test.cl";
        String programSource;
        try {
            programSource = getResourceAsString(filePath);
            sumProgram = CL10.clCreateProgramWithSource(clContext, programSource, errcode_ret);
            System.out.println(programSource);
        } catch (IOException ex) {
            Logger.getLogger(cl_manager.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        CLProgramCallback buildCallback;
        int errcode =
            clBuildProgram(sumProgram, clDevice, "",
                buildCallback =
                    CLProgramCallback.create((program, user_data) -> System.out.println("Building")),
                NULL);
        checkCLError(errcode);

        buildCallback.free();

        // init kernel with constants
        clKernel = clCreateKernel(sumProgram, "sum", errcode_ret);
        checkCLError(errcode_ret);

        System.out.println("create memory");
        createMemory();


        /*
        clSetKernelArg(clKernel, 0, aMemory);
        clSetKernelArg(clKernel, 1, bMemory);
        clSetKernelArg(clKernel, 2, resultMemory);
        clSetKernelArg(clKernel, 3, size);
        */

        clSetKernelArg1p(clKernel, 0, aMemory);
        clSetKernelArg1p(clKernel, 1, bMemory);
        clSetKernelArg1p(clKernel, 2, resultMemory);
        clSetKernelArg1i(clKernel, 3, size);


        final int dimensions = 1;
        PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(dimensions); // In here we put the total number
                                                                                    // of work items we want in each dimension.
        globalWorkSize.put(0, size); // Size is a variable we defined a while back showing how many
                                     // elements are in our arrays.

        // Run the specified number of work units using our OpenCL program kernel
        long startTime = System.nanoTime();
        errcode = clEnqueueNDRangeKernel(clQueue, clKernel, dimensions, null, globalWorkSize, null,
            null, null);

        long endTime = System.nanoTime();
        long timeElapsed = endTime - startTime;
        System.out.println("Execution time in nanoseconds  : " + timeElapsed);

        //clEnqueueReadBuffer(clQueue, answerMem, 1, 0, answer, null, null);
        printResults();

        CL10.clFinish(clQueue);

        cleanup();
        System.out.println("run end");
    }

    private void printResults() {
        // This reads the result memory buffer
        FloatBuffer resultBuff = BufferUtils.createFloatBuffer(size);
        // We read the buffer in blocking mode so that when the method returns we know that the result
        // buffer is full
        clEnqueueReadBuffer(clQueue, resultMemory, true, 0, resultBuff, null, null);
        // Print the values in the result buffer
        for (int i = 0; i < resultBuff.capacity(); i++) {
          System.out.println("result at " + i + " = " + resultBuff.get(i));
        }
        // This should print out 100 lines of result floats, each being 99.
    }
  
  private void createMemory() {
    // Create OpenCL memory object containing the first buffer's list of numbers
    aMemory = CL10.clCreateBuffer(clContext, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_COPY_HOST_PTR,
        getBuffer(), errcode_ret);
    checkCLError(errcode_ret);

    // Create OpenCL memory object containing the second buffer's list of numbers
    bMemory = CL10.clCreateBuffer(clContext, CL10.CL_MEM_WRITE_ONLY | CL10.CL_MEM_COPY_HOST_PTR,
        getBuffer(), errcode_ret);
    checkCLError(errcode_ret);

    // Remember the length argument here is in bytes. 4 bytes per float.
    resultMemory = CL10.clCreateBuffer(clContext, CL10.CL_MEM_READ_ONLY, size * 4, errcode_ret);
    checkCLError(errcode_ret);
  }

  private FloatBuffer getBuffer() {
    // Create float array from 0 to size-1.
    //FloatBuffer aBuff = BufferUtils.createFloatBuffer(size);
    float[] tempData = new float[size];
    for (int i = 0; i < size; i++) {
      tempData[i] = i;
    }
    //aBuff.put(tempData);
    //aBuff.rewind();
    
    FloatBuffer aBuff = toFloatBuffer(tempData);
    return aBuff;
  }


  private void cleanup() {
    // Destroy our kernel and program
    CL10.clReleaseCommandQueue(clQueue);
    CL10.clReleaseKernel(clKernel);
    CL10.clReleaseProgram(sumProgram);
    
    // Destroy our memory objects
    CL10.clReleaseMemObject(aMemory);
    CL10.clReleaseMemObject(bMemory);
    CL10.clReleaseMemObject(resultMemory);
    
    CL.destroy();
  }

  public void initializeCL() {
    errcode_ret = BufferUtils.createIntBuffer(1);
    // Create OpenCL
    // CL.create();
    // Get the first available platform
    try (MemoryStack stack = stackPush()) {
      IntBuffer pi = stack.mallocInt(1);
      checkCLError(clGetPlatformIDs(null, pi));
      if (pi.get(0) == 0) {
        throw new IllegalStateException("No OpenCL platforms found.");
      }

      PointerBuffer platformIDs = stack.mallocPointer(pi.get(0));
      checkCLError(clGetPlatformIDs(platformIDs, (IntBuffer) null));

      for (int i = 0; i < platformIDs.capacity() && i == 0; i++) {
        long platform = platformIDs.get(i);
        clPlatformCapabilities = CL.createPlatformCapabilities(platform);
        clPlatform = platform;
      }
    }


    clDevice = getDevice(clPlatform, clPlatformCapabilities, CL_DEVICE_TYPE_GPU);

    // Create the context
    PointerBuffer ctxProps = BufferUtils.createPointerBuffer(7);
    ctxProps.put(CL_CONTEXT_PLATFORM).put(clPlatform).put(NULL).flip();

    clContext = clCreateContext(ctxProps,
        clDevice, clContextCB = CLContextCallback.create((errinfo, private_info, cb,
            user_data) -> System.out.printf("cl_context_callback\n\tInfo: %s", memUTF8(errinfo))),
        NULL, errcode_ret);

    // create command queue
    clQueue = clCreateCommandQueue(clContext, clDevice, NULL, errcode_ret);
    checkCLError(errcode_ret);
  }

  private static long getDevice(long platform, CLCapabilities platformCaps, int deviceType) {
    try (MemoryStack stack = stackPush()) {
      IntBuffer pi = stack.mallocInt(1);
      checkCLError(clGetDeviceIDs(platform, deviceType, null, pi));

      PointerBuffer devices = stack.mallocPointer(pi.get(0));
      checkCLError(clGetDeviceIDs(platform, deviceType, devices, (IntBuffer) null));

      for (int i = 0; i < devices.capacity(); i++) {
        long device = devices.get(i);

        CLCapabilities caps = CL.createDeviceCapabilities(device, platformCaps);
        if (!(caps.cl_khr_gl_sharing || caps.cl_APPLE_gl_sharing)) {
          continue;
        }

        return device;
      }
    }

    return NULL;
  }
}
