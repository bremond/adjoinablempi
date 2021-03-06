/*
##########################################################
# This file is part of the AdjoinableMPI library         #
# released under the MIT License.                        #
# The full COPYRIGHT notice can be found in the top      #
# level directory of the AdjoinableMPI distribution.     #
########################################################## 
*/

#include <stdlib.h>
#include <mpi.h>

#include "ampi/userIF/activity.h"
#include "ampi/userIF/pairedWith.h"
#include "ampi/userIF/request.h"
#include "ampi/userIF/nt.h"
#include "ampi/adTool/support.h"
#include "ampi/libCommon/modified.h"
#include "ampi/ampi.h"

AMPI_PairedWith pairedWithTable[] =
  {AMPI_TO_RECV, AMPI_FROM_SEND, AMPI_TO_IRECV_WAIT, AMPI_TO_IRECV_WAITALL,
   AMPI_FROM_ISEND_WAIT, AMPI_FROM_ISEND_WAITALL, AMPI_FROM_BSEND, AMPI_FROM_RSEND} ;

MPI_Fint ampi_adouble_precision_;
MPI_Fint ampi_areal_;

void ampi_init_nt_(int* err_code) {
  *err_code = AMPI_Init_NT(0, 0);
}

void ampi_finalize_nt_(int* err_code) {
  *err_code = AMPI_Finalize_NT();}

void ampi_comm_size_(MPI_Fint *commF, int *size, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = MPI_Comm_size(commC, size);
}

void ampi_comm_rank_(MPI_Fint *commF, int *rank, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = MPI_Comm_rank(commC, rank);
}

void adtool_ampi_turn_(double *v, double *vb) {
  ADTOOL_AMPI_Turn(v, vb) ;
}

void fw_ampi_recv_(void* buf,
                   int *count,
                   MPI_Fint *datatypeF,
                   int *src,
                   int *tag,
                   int *pairedWithF,
                   int *commF,
                   int *status,
                   int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Recv(buf, *count, datatype,
                           *src, *tag, pairedWith, commC,
                           (MPI_Status*)status);
}

void bw_ampi_recv_(void* buf,
                   int *count,
                   MPI_Fint *datatypeF,
                   int* src,
                   int* tag,
                   int* pairedWithF,
                   int* commF,
                   int* status,
                   int* err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_AMPI_Recv(buf, *count, datatype,
                           *src, *tag, pairedWith, commC,
                           (MPI_Status*)status);
}

void tls_ampi_recv_(void* buf, void* shadowbuf,
                    int *count,
                    MPI_Fint *datatypeF, MPI_Fint *shadowdatatypeF,
                    int* src,
                    int* tag,
                    int* pairedWithF,
                    int* commF,
                    int* status,
                    int* err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Datatype shadowdatatype = MPI_Type_f2c(*shadowdatatypeF) ;
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Recv(buf, shadowbuf, *count, datatype, shadowdatatype,
                           *src, *tag, pairedWith, commC,
                           (MPI_Status*)status);
}

void fw_ampi_irecv_(void* buf,
                    int *count,
                    MPI_Fint *datatypeF,
                    int *source,
                    int *tag,
                    int *pairedWithF,
                    int *commF,
                    MPI_Fint *requestF,
                    int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Request request   = MPI_Request_f2c(*requestF);
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Irecv(buf, *count, datatype,
                            *source, *tag, pairedWith, commC, &request);
  *requestF = MPI_Request_c2f(request);
}

void bw_ampi_irecv_(void* buf,
                    int *count,
                    MPI_Fint *datatypeF,
                    int *source,
                    int *tag,
                    int *pairedWithF,
                    int *commF,
                    MPI_Fint *requestF,
                    int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Request request   = MPI_Request_f2c(*requestF);
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_AMPI_Irecv(buf, *count, datatype,
                            *source, *tag, pairedWith, commC, &request);
}

void tls_ampi_irecv_(void* buf, void* shadowbuf,
                     int *count,
                     MPI_Fint *datatypeF, MPI_Fint *shadowdatatypeF,
                     int *source,
                     int *tag,
                     int *pairedWithF,
                     int *commF,
                     MPI_Fint *requestF,
                     int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Request request   = MPI_Request_f2c(*requestF);
  MPI_Datatype shadowdatatype = MPI_Type_f2c(*shadowdatatypeF) ;
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Irecv(buf, shadowbuf, *count, datatype, shadowdatatype,
                            *source, *tag, pairedWith, commC, &request);
  *requestF = MPI_Request_c2f(request);
}

void fw_ampi_send_(void* buf, 
                   int *count, 
                   MPI_Fint *datatypeF, 
                   int *dest, 
                   int *tag,
                   int *pairedWithF,
                   int *commF,
                   int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Send (buf, *count, datatype,
                            *dest, *tag, pairedWith, commC);
}

void bw_ampi_send_(void* buf,
                   int *count,
                   MPI_Fint *datatypeF,
                   int *dest, 
                   int *tag,
                   int *pairedWithF,
                   int *commF,
                   int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_AMPI_Send (buf, *count, datatype,
                            *dest, *tag, pairedWith, commC);
}

void tls_ampi_send_(void* buf, void* shadowbuf,
                    int *count,
                    MPI_Fint *datatypeF, MPI_Fint *shadowdatatypeF,
                    int *dest, 
                    int *tag,
                    int *pairedWithF,
                    int *commF,
                    int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Datatype shadowdatatype = MPI_Type_f2c(*shadowdatatypeF) ;
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Send (buf, shadowbuf, *count, datatype, shadowdatatype,
                             *dest, *tag, pairedWith, commC);
}

void fw_ampi_isend_(void* buf,
                    int *count,
                    MPI_Fint *datatypeF,
                    int *dest,
                    int *tag,
                    int *pairedWithF,
                    int *commF,
                    MPI_Fint *requestF,
                    int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Request request   = MPI_Request_f2c(*requestF);
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Isend(buf, *count, datatype,
                            *dest, *tag, pairedWith, commC, &request);
  *requestF = MPI_Request_c2f(request);
}

void bw_ampi_isend_(void* buf,
                    int *count,
                    MPI_Fint *datatypeF,
                    int *dest,
                    int *tag,
                    int *pairedWithF,
                    int *commF,
                    MPI_Fint *requestF,
                    int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Request request   = MPI_Request_f2c(*requestF);
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_AMPI_Isend(buf, *count, datatype,
                            *dest, *tag, pairedWith, commC, &request);
}

void tls_ampi_isend_(void* buf, void* shadowbuf,
                     int *count,
                     MPI_Fint *datatypeF, MPI_Fint *shadowdatatypeF,
                     int *dest,
                     int *tag,
                     int *pairedWithF,
                     int *commF,
                     MPI_Fint *requestF,
                     int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Request request   = MPI_Request_f2c(*requestF);
  MPI_Datatype shadowdatatype = MPI_Type_f2c(*shadowdatatypeF) ;
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Isend(buf, shadowbuf, *count, datatype, shadowdatatype,
                            *dest, *tag, pairedWith, commC, &request);
  *requestF = MPI_Request_c2f(request);
}

void fw_ampi_wait_(MPI_Fint *requestF, MPI_Fint *statusF, int* err_code) {
  MPI_Request request = MPI_Request_f2c( *requestF );
  if( statusF == MPI_F_STATUS_IGNORE ) {
    *err_code = FW_AMPI_Wait( &request,  MPI_STATUS_IGNORE );
  } else if( statusF == MPI_F_STATUSES_IGNORE ) {
    *err_code = FW_AMPI_Wait( &request,  MPI_STATUSES_IGNORE );
  } else {
    MPI_Status status;
    MPI_Status_f2c( statusF, &status );
    *err_code = FW_AMPI_Wait( &request,  &status );
    MPI_Status_c2f( &status, statusF ) ;
  }
}

void bw_ampi_wait_(MPI_Fint *requestF, MPI_Fint *statusF, int* err_code) {
  MPI_Request request = MPI_Request_f2c( *requestF );
  *err_code = BW_AMPI_Wait( &request,  MPI_STATUS_IGNORE );
  *requestF = MPI_Request_c2f(request);
}

void tls_ampi_wait_(MPI_Fint *requestF, MPI_Fint *statusF, int* err_code) {
  MPI_Request request = MPI_Request_f2c( *requestF );
  if( statusF == MPI_F_STATUS_IGNORE ) {
    *err_code = TLS_AMPI_Wait( &request,  MPI_STATUS_IGNORE );
  }
  else if( statusF == MPI_F_STATUSES_IGNORE ) {
    *err_code = TLS_AMPI_Wait( &request,  MPI_STATUSES_IGNORE );
  }
  else {
    MPI_Status status;
    MPI_Status_f2c( statusF, &status );
    *err_code = TLS_AMPI_Wait( &request,  &status );
    MPI_Status_c2f( &status, statusF ) ;
  }
}

void fw_ampi_barrier_(int *commF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Barrier(commC) ;
}

void bw_ampi_barrier_(int *commF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_AMPI_Barrier(commC) ;
}

void tls_ampi_barrier_(int *commF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Barrier(commC) ;
}

void fw_ampi_gather_(void* sbuf, int *scount, MPI_Fint *stypeF, void* rbuf, int *rcount, MPI_Fint *rtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Gather(sbuf, *scount, stype, rbuf, *rcount, rtype, *root, commC) ;
}

void bw_ampi_gather_(void* sbuf, int *scount, MPI_Fint *stypeF, void* rbuf, int *rcount, MPI_Fint *rtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_AMPI_Gather(sbuf, *scount, stype, rbuf, *rcount, rtype, *root, commC) ;
}

void tls_ampi_gather_(void* sbuf, void* shadowsbuf, int *scount, MPI_Fint *stypeF, MPI_Fint *shadowstypeF, void* rbuf, void* shadowrbuf, int *rcount, MPI_Fint *rtypeF, MPI_Fint *shadowrtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype shadowstype = MPI_Type_f2c(*shadowstypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Datatype shadowrtype = MPI_Type_f2c(*shadowrtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Gather(sbuf, shadowsbuf, *scount, stype, shadowstype, rbuf, shadowrbuf, *rcount, rtype, shadowrtype, *root, commC) ;
}

void fw_ampi_scatter_(void* sbuf, int *scount, MPI_Fint *stypeF, void* rbuf, int *rcount, MPI_Fint *rtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Scatter(sbuf, *scount, stype, rbuf, *rcount, rtype, *root, commC) ;
}

void bw_ampi_scatter_(void* sbuf, int *scount, MPI_Fint *stypeF, void* rbuf, int *rcount, MPI_Fint *rtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_AMPI_Scatter(sbuf, *scount, stype, rbuf, *rcount, rtype, *root, commC) ;
}

void tls_ampi_scatter_(void* sbuf, void* shadowsbuf, int *scount, MPI_Fint *stypeF, MPI_Fint *shadowstypeF, void* rbuf, void* shadowrbuf, int *rcount, MPI_Fint *rtypeF, MPI_Fint *shadowrtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype shadowstype = MPI_Type_f2c(*shadowstypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Datatype shadowrtype = MPI_Type_f2c(*shadowrtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Scatter(sbuf, shadowsbuf, *scount, stype, shadowstype, rbuf, shadowrbuf, *rcount, rtype, shadowrtype, *root, commC) ;
}

void fw_ampi_allgather_(void* sbuf, int *scount, MPI_Fint *stypeF, void* rbuf, int *rcount, MPI_Fint *rtypeF, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Allgather(sbuf, *scount, stype, rbuf, *rcount, rtype, commC) ;
}

void bw_ampi_allgather_(void* sbuf, int *scount, MPI_Fint *stypeF, void* rbuf, int *rcount, MPI_Fint *rtypeF, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_AMPI_Allgather(sbuf, *scount, stype, rbuf, *rcount, rtype, commC) ;
}

void tls_ampi_allgather_(void* sbuf, void* shadowsbuf, int *scount, MPI_Fint *stypeF, MPI_Fint *shadowstypeF, void* rbuf, void* shadowrbuf, int *rcount, MPI_Fint *rtypeF, MPI_Fint *shadowrtypeF, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype shadowstype = MPI_Type_f2c(*shadowstypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Datatype shadowrtype = MPI_Type_f2c(*shadowrtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Allgather(sbuf, shadowsbuf, *scount, stype, shadowstype, rbuf, shadowrbuf, *rcount, rtype, shadowrtype, commC) ;
}

void fw_ampi_gatherv_(void* sbuf, int *scount, MPI_Fint *stypeF, void* rbuf, int* rcounts, int *displs, MPI_Fint *rtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Gatherv(sbuf, *scount, stype, rbuf, rcounts, displs, rtype, *root, commC) ;
}

void bw_ampi_gatherv_(void* sbuf, int *scount, MPI_Fint *stypeF, void* rbuf, int *rcounts, int *displs, MPI_Fint *rtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_AMPI_Gatherv(sbuf, *scount, stype, rbuf, rcounts, displs, rtype, *root, commC) ;
}

void tls_ampi_gatherv_(void* sbuf, void* shadowsbuf, int *scount, MPI_Fint *stypeF, MPI_Fint *shadowstypeF, void* rbuf, void* shadowrbuf, int *rcounts, int *displs, int *shadowdispls, MPI_Fint *rtypeF, MPI_Fint *shadowrtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype shadowstype = MPI_Type_f2c(*shadowstypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Datatype shadowrtype = MPI_Type_f2c(*shadowrtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Gatherv(sbuf, shadowsbuf, *scount, stype, shadowstype, rbuf, shadowrbuf, rcounts, displs, shadowdispls, rtype, shadowrtype, *root, commC) ;
}

void fw_ampi_scatterv_(void* sbuf, int *scounts, int *displs, MPI_Fint *stypeF, void* rbuf, int *rcount, MPI_Fint *rtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Scatterv(sbuf, scounts, displs, stype, rbuf, *rcount, rtype, *root, commC) ;
}

void bw_ampi_scatterv_(void* sbuf, int *scounts, int *displs, MPI_Fint *stypeF, void* rbuf, int *rcount, MPI_Fint *rtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_AMPI_Scatterv(sbuf, scounts, displs, stype, rbuf, *rcount, rtype, *root, commC) ;
}

void tls_ampi_scatterv_(void* sbuf, void* shadowsbuf, int *scounts, int *displs, int *shadowdispls, MPI_Fint *stypeF, MPI_Fint *shadowstypeF, void* rbuf, void* shadowrbuf, int *rcount, MPI_Fint *rtypeF, MPI_Fint *shadowrtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype shadowstype = MPI_Type_f2c(*shadowstypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Datatype shadowrtype = MPI_Type_f2c(*shadowrtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Scatterv(sbuf, shadowsbuf, scounts, displs, shadowdispls, stype, shadowstype, rbuf, shadowrbuf, *rcount, rtype, shadowrtype, *root, commC) ;
}

void fw_ampi_allgatherv_(void* sbuf, int *scount, MPI_Fint *stypeF, void* rbuf, int *rcounts, int *displs, MPI_Fint *rtypeF, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Allgatherv(sbuf, *scount, stype, rbuf, rcounts, displs, rtype, commC) ;
}

void bw_ampi_allgatherv_(void* sbuf, int *scount, MPI_Fint *stypeF, void* rbuf, int *rcounts, int *displs, MPI_Fint *rtypeF, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_AMPI_Allgatherv(sbuf, *scount, stype, rbuf, rcounts, displs, rtype, commC) ;
}

void tls_ampi_allgatherv_(void* sbuf, void* shadowsbuf, int *scount, MPI_Fint *stypeF, MPI_Fint *shadowstypeF, void* rbuf, void* shadowrbuf, int *rcounts, int *displs, int *shadowdispls, MPI_Fint *rtypeF, MPI_Fint *shadowrtypeF, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype shadowstype = MPI_Type_f2c(*shadowstypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Datatype shadowrtype = MPI_Type_f2c(*shadowrtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Allgatherv(sbuf, shadowsbuf, *scount, stype, shadowstype, rbuf, shadowrbuf, rcounts, displs, shadowdispls, rtype, shadowrtype, commC) ;
}

void fw_ampi_bcast_(void* buf, int *count, MPI_Fint *typeF, int *root, int *commF, int* err_code) {
  MPI_Datatype type = MPI_Type_f2c(*typeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Bcast(buf, *count, type, *root, commC) ;
}

void bw_ampi_bcast_(void* buf, int *count, MPI_Fint *typeF, int *root, int *commF, int* err_code) {
  MPI_Datatype type = MPI_Type_f2c(*typeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_AMPI_Bcast(buf, *count, type, *root, commC) ;
}

void tls_ampi_bcast_(void* buf, void* shadowbuf, int *count, MPI_Fint *typeF, MPI_Fint *shadowtypeF, int *root, int *commF, int* err_code) {
  MPI_Datatype type = MPI_Type_f2c(*typeF) ;
  MPI_Datatype shadowtype = MPI_Type_f2c(*shadowtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Bcast(buf, shadowbuf, *count, type, shadowtype, *root, commC) ;
}

void fw_ampi_reduce_(void* sbuf, void* rbuf,
                      int *count,
                      MPI_Fint *datatypeF,
                      MPI_Fint *opF,
                      int *root,
                      int *commF,
                      int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Op op = MPI_Op_f2c(*opF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Reduce(sbuf, rbuf, *count, datatype,
                              op, *root, commC) ;
}

void bws_ampi_reduce_(void* sbuf, void* sbufb,
                      void* rbuf, void* rbufb,
                      int *count,
                      MPI_Fint *datatypeF, MPI_Fint *datatypebF,
                      MPI_Fint *opF, void* uopbF,
                      int *root,
                      int *commF,
                      int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Datatype datatypeb = MPI_Type_f2c(*datatypebF) ;
  MPI_Op op = MPI_Op_f2c(*opF) ;
  TLM_userFunctionF* uopb = 0 /*???(uopbF)*/ ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BWS_AMPI_Reduce(sbuf, sbufb,
                              rbuf, rbufb,
                              *count,
                              datatype, datatypeb,
                              op, uopb,
                              *root, commC) ;
}

void tls_ampi_reduce_(void* sbuf, void* shadowsbuf,
                      void* rbuf, void* shadowrbuf,
                      int *count,
                      MPI_Fint *datatypeF, MPI_Fint *shadowdatatypeF,
                      MPI_Fint *opF, void* uopdF,
                      int *root,
                      int *commF,
                      int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Datatype shadowdatatype = MPI_Type_f2c(*shadowdatatypeF) ;
  MPI_Op op = MPI_Op_f2c(*opF) ;
  TLM_userFunctionF* uopd = 0 /*???(uopdF)*/ ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Reduce(sbuf, shadowsbuf,
                              rbuf, shadowrbuf,
                              *count,
                              datatype, shadowdatatype,
                              op, uopd,
                              *root, commC) ;
}

void fw_ampi_allreduce_(void* sbuf, void* rbuf,
                      int *count,
                      MPI_Fint *datatypeF,
                      MPI_Fint *opF,
                      int *commF,
                      int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Op op = MPI_Op_f2c(*opF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_AMPI_Allreduce(sbuf, rbuf, *count, datatype,
                              op, commC) ;
}

void bws_ampi_allreduce_(void* sbuf, void* sbufb,
                      void* rbuf, void* rbufb,
                      int *count,
                      MPI_Fint *datatypeF, MPI_Fint *datatypebF,
                      MPI_Fint *opF, void* uopbF,
                      int *commF,
                      int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Datatype datatypeb = MPI_Type_f2c(*datatypebF) ;
  MPI_Op op = MPI_Op_f2c(*opF) ;
  TLM_userFunctionF* uopb = 0 /*???(uopbF)*/ ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BWS_AMPI_Allreduce(sbuf, sbufb,
                              rbuf, rbufb,
                              *count,
                              datatype, datatypeb,
                              op, uopb,
                              commC) ;
}

void tls_ampi_allreduce_(void* sbuf, void* shadowsbuf,
                      void* rbuf, void* shadowrbuf,
                      int *count,
                      MPI_Fint *datatypeF, MPI_Fint *shadowdatatypeF,
                      MPI_Fint *opF, void* uopdF,
                      int *commF,
                      int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Datatype shadowdatatype = MPI_Type_f2c(*shadowdatatypeF) ;
  MPI_Op op = MPI_Op_f2c(*opF) ;
  TLM_userFunctionF* uopd = 0 /*???(uopdF)*/ ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Allreduce(sbuf, shadowsbuf,
                              rbuf, shadowrbuf,
                              *count,
                              datatype, shadowdatatype,
                              op, uopd,
                              commC) ;
}

void ampi_comm_dup_(MPI_Fint *commF, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  *err_code = MPI_Comm_dup(commC, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void ampi_comm_dup_nt_(MPI_Fint *commF, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  *err_code = MPI_Comm_dup(commC, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void ampi_comm_split_(MPI_Fint *commF, int *color, int *key, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  *err_code = MPI_Comm_split(commC, *color, *key, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void ampi_comm_split_nt_(MPI_Fint *commF, int *color, int *key, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  *err_code = MPI_Comm_split(commC, *color, *key, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void ampi_comm_create_(MPI_Fint *commF, MPI_Fint *groupF, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  MPI_Group group = MPI_Group_f2c(*groupF) ;
  *err_code = MPI_Comm_create(commC, group, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void ampi_comm_create_nt_(MPI_Fint *commF, MPI_Fint *groupF, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  MPI_Group group = MPI_Group_f2c(*groupF) ;
  *err_code = MPI_Comm_create(commC, group, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void ampi_comm_free_(MPI_Fint *commF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = MPI_Comm_free(&commC) ;
}

void ampi_comm_free_nt_(MPI_Fint *commF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = MPI_Comm_free(&commC) ;
}

void tls_ampi_comm_dup_(MPI_Fint *commF, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  *err_code = TLS_AMPI_Comm_dup(commC, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void tls_ampi_comm_dup_nt_(MPI_Fint *commF, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  *err_code = TLS_AMPI_Comm_dup_NT(commC, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void tls_ampi_comm_split_(MPI_Fint *commF, int *color, int *key, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  *err_code = TLS_AMPI_Comm_split(commC, *color, *key, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void tls_ampi_comm_split_nt_(MPI_Fint *commF, int *color, int *key, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  *err_code = TLS_AMPI_Comm_split_NT(commC, *color, *key, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void tls_ampi_comm_create_(MPI_Fint *commF, MPI_Fint *groupF, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  MPI_Group group = MPI_Group_f2c(*groupF) ;
  *err_code = TLS_AMPI_Comm_create(commC, group, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void tls_ampi_comm_create_nt_(MPI_Fint *commF, MPI_Fint *groupF, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  MPI_Group group = MPI_Group_f2c(*groupF) ;
  *err_code = TLS_AMPI_Comm_create_NT(commC, group, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void tls_ampi_comm_free_(MPI_Fint *commF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Comm_free(&commC) ;
}

void tls_ampi_comm_free_nt_(MPI_Fint *commF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_AMPI_Comm_free_NT(&commC) ;
}
