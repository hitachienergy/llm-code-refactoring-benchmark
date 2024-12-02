// libs/client/GarnetClient.cs
// 579/677
// https://github.com/microsoft/garnet
// c#
        async ValueTask InternalExecuteAsync(Memory<byte> op, Memory<byte> clusterOp, string nodeId, long currentAddress, long nextAddress, long payloadPtr, int payloadLength, CancellationToken token = default)
        {
            Debug.Assert(nodeId != null);

            int totalLen = 0;
            int arraySize = 1;

            totalLen += op.Length;

            int len = clusterOp.Length;
            totalLen += 1 + NumUtils.NumDigits(len) + 2 + len + 2;
            arraySize++;

            len = Encoding.UTF8.GetByteCount(nodeId);
            totalLen += 1 + NumUtils.NumDigits(len) + 2 + len + 2;
            arraySize++;

            len = NumUtils.NumDigitsInLong(currentAddress);
            totalLen += 1 + NumUtils.NumDigits(len) + 2 + len + 2;
            arraySize++;

            len = NumUtils.NumDigitsInLong(nextAddress);
            totalLen += 1 + NumUtils.NumDigits(len) + 2 + len + 2;
            arraySize++;

            len = payloadLength;
            totalLen += 1 + NumUtils.NumDigits(len) + 2 + len + 2;
            arraySize++;

            totalLen += 1 + NumUtils.NumDigits(arraySize) + 2;

            if (totalLen > networkWriter.PageSize)
            {
                ThrowException(new Exception($"Entry of size {totalLen} does not fit on page of size {networkWriter.PageSize}. Try increasing sendPageSize parameter to GarnetClient constructor."));
            }

            // No need for gate as this is a void return
            // await InputGateAsync(token);

            try
            {
                networkWriter.epoch.Resume();

                #region reserveSpaceAndWriteIntoNetworkBuffer
                int taskId;
                long address;
                while (true)
                {
                    token.ThrowIfCancellationRequested();
                    if (!IsConnected)
                    {
                        Dispose();
                        ThrowException(disposeException);
                    }
                    (taskId, address) = networkWriter.TryAllocate(totalLen, out var flushEvent);
                    if (address >= 0) break;
                    try
                    {
                        networkWriter.epoch.Suspend();
                        await flushEvent.WaitAsync(token).ConfigureAwait(false);
                    }
                    finally
                    {
                        networkWriter.epoch.Resume();
                    }
                }

                unsafe
                {
                    byte* curr = (byte*)networkWriter.GetPhysicalAddress(address);
                    byte* end = curr + totalLen;
                    RespWriteUtils.WriteArrayLength(arraySize, ref curr, end);

                    RespWriteUtils.WriteDirect(op.Span, ref curr, end);
                    RespWriteUtils.WriteBulkString(clusterOp.Span, ref curr, end);
                    RespWriteUtils.WriteUtf8BulkString(nodeId, ref curr, end);
                    RespWriteUtils.WriteArrayItem(currentAddress, ref curr, end);
                    RespWriteUtils.WriteArrayItem(nextAddress, ref curr, end);
                    RespWriteUtils.WriteBulkString(new Span<byte>((void*)payloadPtr, payloadLength), ref curr, end);

                    Debug.Assert(curr == end);
                }
                #endregion

                if (!IsConnected)
                {
                    Dispose();
                    ThrowException(disposeException);
                }
                // Console.WriteLine($"Filled {address}-{address + totalLen}");
                networkWriter.epoch.ProtectAndDrain();
                networkWriter.DoAggressiveShiftReadOnly();
            }
            finally
            {
                networkWriter.epoch.Suspend();
            }
            return;
        }