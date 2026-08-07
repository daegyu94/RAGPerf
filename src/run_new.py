def main():
    import os, sys
    import utils.python_utils as pyutils
    import time
    import itertools

    if not any(
        [p in arg for p in ["--log_dir", "--create_log_dir"] for arg in sys.argv]
    ):
        sys.argv.append(
            f"--log_dir={os.path.join(pyutils.get_script_dir(__file__), 'output')}"
        )
        sys.argv.append(f"--create_log_dir=True")

    from utils.logger import logging, Logger, log_time_breakdown, save_config_to_log_dir

    from config import load_config, get_db_collection_name
    from utils.python_utils import get_by_path
    import utils.colored_print as cprint

    # put those before any other imports to prevent loading wrong libstdc++.so
    from monitoring_sys.config_parser.msys_config_parser import (
        StaticEnv,
        MacroTranslator,
    )
    from monitoring_sys import MSys
    from monitoring_sys.config_parser.msys_config_parser import MSysConfig

    import torch
    import argparse
    import pickle
    import _pickle as cPickle

    from vectordb.milvus_api import milvus_client

    from datasetLoader.AudioDatasetLoader import AudioDatasetLoader
    from encoder.AudioEncoder import AudioEncoder
    from RAGPipeline.AudioRAGPipeline import AudioRAGPipeline

    # avoid warning about TOKENIZERS_PARALLELISM
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    output_path = Logger().log_dirpath
    cprint.iprintf(f"Using output path: {output_path}")

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--msys-config",
        type=str,
        help="Path to the monitoring system configuration file",
    )
    # parser.add_argument("-d", "--dry_run", action="store_true", help="Run in dry run mode, no actual processing")
    args = parser.parse_known_args()[0]
    if not args.config:
        raise ValueError("Please provide a configuration file using --config")
    config = load_config(args.config)

    if not args.msys_config:
        raise ValueError(
            "Please provide a monitoring system configuration file using --msys-config"
        )
    with open(args.msys_config, "r") as fin:
        translated_config = (
            MacroTranslator(StaticEnv.get_static_env("global")).translate(fin).read()
        )
    with open(os.path.join(output_path, "translated_msys_config.yaml"), "w") as fout:
        fout.write(translated_config)
    monitor = MSys(MSysConfig.from_yaml_string(translated_config))
    monitor.report_status(verbose=False, detail=True)

    # set collection name
    if not config["sys"]["vector_db"]["collection_name"] == "":
        collection_name = get_db_collection_name(
            config["sys"]["vector_db"]["collection_name"]
        )
    else:
        collection_name = get_db_collection_name(f"{config['run_name']}")
    cprint.iprintf(f"*** Start the run with collection {collection_name}")

    # set db
    if config["sys"]["vector_db"]["type"] == "milvus":
        db_client = milvus_client(
            db_path=config["sys"]["vector_db"]["db_path"],
            db_token=config["sys"]["vector_db"]["db_token"],
            collection_name=collection_name,
            drop_previous_collection=config["sys"]["vector_db"][
                "drop_previous_collection"
            ],
            # dim=config["sys"]["vector_db"]["dim"],
            index_type=config["rag"]["build_index"]["index_type"],
            metric_type=config["rag"]["build_index"]["metric_type"],
            trace=config["sys"]["vector_db"].get("trace"),
        )
    elif config["sys"]["vector_db"]["type"] == "lancedb":
        from vectordb.lancedb_api import lance_client

        db_client = lance_client(
            db_path=config["sys"]["vector_db"]["db_path"],
            collection_name=collection_name,
            # dim=config["sys"]["vector_db"]["dim"],
            index_type=config["rag"]["build_index"]["index_type"],
            metric_type=config["rag"]["build_index"]["metric_type"],
            drop_previous_collection=config["sys"]["vector_db"][
                "drop_previous_collection"
            ],
        )
    elif config["sys"]["vector_db"]["type"] == "qdrant":
        from vectordb.qdrant_api import qdrant_client

        db_client = qdrant_client(
            db_path=config["sys"]["vector_db"]["db_path"],
            collection_name=collection_name,
            # dim=config["sys"]["vector_db"]["dim"],
            index_type=config["rag"]["build_index"]["index_type"],
            metric_type=config["rag"]["build_index"]["metric_type"],
            drop_previous_collection=config["sys"]["vector_db"][
                "drop_previous_collection"
            ],
        )
    elif config["sys"]["vector_db"]["type"] == "chroma":
        from vectordb.chroma_api import chroma_client

        db_client = chroma_client(
            db_path=config["sys"]["vector_db"]["db_path"],
            collection_name=collection_name,
            # dim=config["sys"]["vector_db"]["dim"],
            index_type=config["rag"]["build_index"]["index_type"],
            metric_type=config["rag"]["build_index"]["metric_type"],
            drop_previous_collection=config["sys"]["vector_db"][
                "drop_previous_collection"
            ],
        )
    elif config["sys"]["vector_db"]["type"] == "elasticsearch":
        from vectordb.elastic_api import elastic_client

        db_client = elastic_client(
            db_path=config["sys"]["vector_db"]["db_path"],
            collection_name=collection_name,
            # dim=config["sys"]["vector_db"]["dim"],
            index_type=config["rag"]["build_index"]["index_type"],
            metric_type=config["rag"]["build_index"]["metric_type"],
            drop_previous_collection=config["sys"]["vector_db"][
                "drop_previous_collection"
            ],
        )
    else:
        raise ValueError(
            f"Unsupported vector database type: {config['sys']['vector_db']['type']}"
        )

    db_client.setup()
    cprint.iprintf(f"*** Vector DB setup done")

    # prepare workload
    dataset_name = config["bench"]["dataset"]
    save_config_to_log_dir(args.config)
    if config["bench"].get("type") == "audio":
        actions = config["rag"]["action"]
        audio_config = config["rag"].get("audio", {})
        loader = AudioDatasetLoader(
            dataset_name=dataset_name,
            dataset_config=audio_config.get("dataset_config", "clean"),
            split=audio_config.get("split", "train.100"),
            streaming=audio_config.get("streaming", False),
            cache_dir=audio_config.get("cache_dir"),
        )
        count = audio_config.get("sample_count")
        if count is None:
            if loader.total_length is None:
                raise ValueError(
                    "rag.audio.sample_count is required when rag.audio.streaming is true"
                )
            count = max(
                1,
                int(
                    loader.total_length
                    * config["bench"]["preprocessing"].get("dataset_ratio", 1.0)
                ),
            )
        batch_size = int(
            audio_config.get(
                "batch_size", config["rag"].get("embedding", {}).get("batch_size", 8)
            )
        )
        encoder_kwargs = {
            "asr_model": audio_config.get("asr_model", "openai/whisper-tiny"),
            "embedding_model": audio_config.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "device": audio_config.get(
                "device", config["rag"].get("embedding", {}).get("device", "cpu")
            ),
            "batch_size": batch_size,
        }
        needs_corpus = actions.get("insert", False)
        if needs_corpus and not actions.get("embedding", False):
            raise ValueError(
                "Audio RAG insert requires rag.action.embedding=true; "
                "precomputed audio vectors are not supported"
            )
        if needs_corpus:
            encoder = AudioEncoder(**encoder_kwargs)
            encoder.load_encoder()
            collection_created = db_client.has_collection(collection_name)
            inserted = 0
            try:
                iterator = loader.iter_samples(limit=int(count))
                while True:
                    samples = list(itertools.islice(iterator, batch_size))
                    if not samples:
                        break
                    vectors, transcripts = encoder.embedding_with_text(
                        [sample["audio"] for sample in samples]
                    )
                    rows = [
                        {
                            "vector": vector,
                            "text": transcript,
                            "metadata": {
                                "dataset": dataset_name,
                                **sample.get("metadata", {}),
                            },
                        }
                        for sample, vector, transcript in zip(
                            samples, vectors, transcripts
                        )
                    ]
                    if not collection_created:
                        db_client.create_collection(
                            collection_name=collection_name,
                            dim=len(vectors[0]),
                            auto_id=True,
                        )
                        collection_created = True
                    db_client.insert_data(
                        rows,
                        collection_name=collection_name,
                        insert_batch_size=len(rows),
                        create_collection=False,
                    )
                    inserted += len(rows)
            finally:
                encoder.free_encoder()
            if inserted == 0:
                raise ValueError("Audio dataset produced no samples")
            if actions.get("build_index", False):
                db_client.build_index(
                    collection_name=collection_name,
                    index_type=config["rag"]["build_index"]["index_type"],
                    metric_type=config["rag"]["build_index"]["metric_type"],
                )

        if actions.get("retrieval", False) or actions.get("generation", False):
            query_loader = AudioDatasetLoader(
                dataset_name=dataset_name,
                dataset_config=audio_config.get("dataset_config", "clean"),
                split=audio_config.get("split", "train.100"),
                streaming=audio_config.get("streaming", False),
                cache_dir=audio_config.get("cache_dir"),
            )
            query_count = min(
                int(count),
                int(config["rag"].get("retrieval", {}).get("question_num", count)),
            )
            pipeline = AudioRAGPipeline(
                encoder=AudioEncoder(**encoder_kwargs),
                client=db_client,
                collection_name=collection_name,
                top_k=config["rag"].get("retrieval", {}).get("top_k", 5),
                retrieval_batch_size=config["rag"]
                .get("retrieval", {})
                .get("retrieval_batch_size", 1),
            )
            pipeline.retrieve(
                (
                    sample["audio"]
                    for sample in query_loader.iter_samples(limit=query_count)
                )
            )
        if hasattr(db_client, "close_trace"):
            db_client.close_trace()
        monitor.close()
        return
    # for image RAG
    if config["bench"]["type"] == "image":
        from datasetLoader.PDFDatasetLoader import PDFDatasetLoader
        from datasetPreprocess.PDFDatasetPreprocess import PDFDatasetPreprocess
        from RAGRequest.TextsRAGRequest import WikipediaRequests
        from RAGPipeline.retriever.BaseRetriever import BaseRetriever
        from encoder.ColPaliEncoder import ColPaliEncoder

        pass
        # preprocess dataset
        with monitor:
            if config["rag"]["action"]["preprocess"]:
                log_time_breakdown("start")
                if dataset_name == "common-pile/arxiv_papers":
                    cprint.iprintf(
                        f"*** Start loading dataset: {dataset_name}, time : {time.monotonic_ns()} "
                    )
                    dataset_ratio = config["bench"]["preprocessing"]["dataset_ratio"]
                    loader = PDFDatasetLoader(dataset_name=dataset_name)
                    samples_length = int(loader.total_length * dataset_ratio)
                    loader.download_pdf(load_num=samples_length)
                    df = loader.get_dataset_slice(length=samples_length, offset=0)
                    cprint.iprintf(
                        f"*** Done Loaded dataset: {dataset_name}, total samples: {len(df)}, done"
                    )
                log_time_breakdown("chunking")
                chunker = PDFDatasetPreprocess()
                pages = chunker.chunking_PDF_to_image(df)

            # embedding
            if config["rag"]["action"]["embedding"]:
                cprint.iprintf(
                    f"*** Start embedding images, time : {time.monotonic_ns()}"
                )
                log_time_breakdown("embed")
                embedder = ColPaliEncoder(
                    device=config["rag"]["embedding"]["device"],
                    model_name=config["rag"]["embedding"]["sentence_transformers_name"],
                    embedding_batch_size=config["rag"]["embedding"]["batch_size"],
                )
                embedder.load_encoder()
                dict_list = embedder.embedding(pages)
                embedder.free_encoder()
                print(
                    f"***Embedding done, total {len(dict_list)} embeddings, time : {time.monotonic_ns()}"
                )

            if config["rag"]["action"]["insert"]:
                print(
                    f"***Start inserting embeddings into collection: {collection_name}, time : {time.monotonic_ns()}"
                )
                log_time_breakdown("insert")
                if config["sys"]["vector_db"]["type"] == "lancedb":
                    db_client.create_collection(
                        collection_name=collection_name,
                        dim=len(dict_list[0]["vector"]),
                        data_type="image",
                    )

                db_client.insert_data(
                    dict_list=dict_list,
                    collection_name=collection_name,
                    insert_batch_size=config["rag"]["insert"]["batch_size"],
                    create_collection=True,
                )
                print(
                    f"***Insertion done, total {len(dict_list)} embeddings inserted, time : {time.monotonic_ns()}"
                )
                log_time_breakdown("done")
            if config["rag"]["action"]["build_index"]:
                db_client.build_index(
                    collection_name=collection_name,
                    index_type=config["rag"]["build_index"]["index_type"],
                    metric_type=config["rag"]["build_index"]["metric_type"],
                )
                print(f"***Indexing done for collection: {collection_name}")
        if config["rag"]["action"]["generation"] == True:
            RAGRequest = WikipediaRequests(
                run_name=config["run_name"],
                collection_name=collection_name,
                req_type="query",
                req_count=config["rag"]["retrieval"]["question_num"],
            )
            print(f"***End request preparation")

            # prepare pipeline
            retriever = BaseRetriever(
                collection_name=collection_name,
                top_k=config["rag"]["retrieval"]["top_k"],
                retrieval_batch_size=config["rag"]["retrieval"]["retrieval_batch_size"],
                client=db_client,
            )
            from RAGPipeline.ImageRAGPipline import ImagesRAGPipeline
            from RAGPipeline.responser.ImagesResponser import ImageResponser

            responser = ImageResponser(
                model=config["rag"]["generation"]["model"],
                device=config["rag"]["generation"]["device"],
            )
            embedder = ColPaliEncoder(
                device=config["rag"]["embedding"]["device"],
                model_name=config["rag"]["embedding"]["sentence_transformers_name"],
                embedding_batch_size=config["rag"]["embedding"]["batch_size"],
            )
            RAGPipline = ImagesRAGPipeline(
                retriever=retriever,
                responser=responser,
                embedder=embedder,
            )

            # pipeline.check()
            import utils.colored_print as cprint

            with monitor:
                RAGPipline.process(
                    RAGRequest,
                    batch_size=config["rag"]["pipeline"]["batch_size"],
                )
        if hasattr(db_client, "close_trace"):
            db_client.close_trace()
        monitor.close()

        return
    elif config["bench"]["type"] == "text":
        from datasetLoader.TextDatasetLoader import TextDatasetLoader
        from datasetPreprocess.TextDatasetPreprocess import TextDatasetPreprocess
        from datasetLoader.PDFDatasetLoader import PDFDatasetLoader
        from datasetPreprocess.PDFDatasetPreprocess import PDFDatasetPreprocess
        from RAGRequest.TextsRAGRequest import WikipediaRequests
        from RAGPipeline.retriever.BaseRetriever import BaseRetriever
        from RAGPipeline.reranker.CrossEncoderReranker import CrossEncoderReranker
        from encoder.sentenceTransformerEncoder import SentenceTransformerEncoder

        # preprocess dataset
        if config["rag"]["action"]["preprocess"]:
            # if True:
            log_time_breakdown("start")
            with monitor:
                # TODO: add length and offset into config
                # download and load dataset
                # if config["rag"]["action"]["preprocess"]:
                if dataset_name == "wikimedia/wikipedia":
                    dataset_ratio = config["bench"]["preprocessing"]["dataset_ratio"]
                    loader = TextDatasetLoader(dataset_name=dataset_name)
                    samples_length = int(loader.total_length * dataset_ratio)
                    df = loader.get_dataset_slice(length=samples_length, offset=0)
                    cprint.iprintf(
                        f"*** Done Loaded dataset: {dataset_name}, total samples: {len(df)}, done"
                    )
                elif dataset_name == "common-pile/arxiv_papers":
                    dataset_ratio = config["bench"]["preprocessing"]["dataset_ratio"]
                    loader = PDFDatasetLoader(dataset_name=dataset_name)
                    samples_length = int(loader.total_length * dataset_ratio)
                    loader.download_pdf(load_num=samples_length)
                    df = loader.get_dataset_slice(length=samples_length, offset=0)
                    cprint.iprintf(
                        f"*** Done Loaded dataset: {dataset_name}, total samples: {len(df)}, done"
                    )
                # chunking datasets
                if dataset_name == "wikimedia/wikipedia":
                    chunker = TextDatasetPreprocess(
                        chunk_size=config["bench"]["preprocessing"]["chunk_size"],
                        chunk_overlap=config["bench"]["preprocessing"]["chunk_overlap"],
                    )
                    log_time_breakdown("chunking")
                    chunked_texts = chunker.chunking_text_to_text(df)
                    cprint.iprintf(
                        f"*** Chunking done, total {len(chunked_texts)} chunks"
                    )
                elif dataset_name == "common-pile/arxiv_papers":
                    chunker = PDFDatasetPreprocess()
                    log_time_breakdown(
                        "convert"
                    )  # todo separate chunking and converting
                    docs = chunker.convert_PDF_to_text(df)
                    log_time_breakdown("chunking")
                    chunked_texts = chunker.chunking_PDF_to_text(docs)
                    cprint.iprintf(
                        f"*** Chunking done, total {len(chunked_texts)} chunks"
                    )

                embeddings_dim = None
                # embedding
                if config["rag"]["action"]["embedding"]:
                    cprint.iprintf(f"*** Start embedding texts")
                    log_time_breakdown("embed")
                    embedder = SentenceTransformerEncoder(
                        device=config["rag"]["embedding"]["device"],
                        sentence_transformers_name=config["rag"]["embedding"][
                            "sentence_transformers_name"
                        ],
                        embedding_batch_size=config["rag"]["embedding"]["batch_size"],
                    )
                    embedder.load_encoder()
                    embeddings_dim = embedder.dim
                    embeddings = embedder.embedding(chunked_texts)
                    embedder.free_encoder()
                    print(f"***Embedding done, total {len(embeddings)} embeddings")
                    if config["rag"]["embedding"]["store"] == True:
                        store_path = config["rag"]["embedding"]["filepath"]
                        # Store data
                        os.makedirs(os.path.dirname(store_path), exist_ok=True)
                        with open(store_path, "wb") as handle:
                            pickle.dump(
                                embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL
                            )

                if config["rag"]["embedding"]["load"] == True:
                    log_time_breakdown("load")
                    load_path = config["rag"]["embedding"]["filepath"]
                    with open(load_path, "rb") as handle:
                        embeddings = cPickle.load(handle)
                    print(f"***Embedding loaded, total {len(embeddings)} embeddings")
                    # print(f"***Embedding example0: {embeddings[0]['vector']}")
                    # print(f"***Embedding example0: {embeddings[0]}")
                    # print(f"***Embedding dim: {len(embeddings[0]['vector'])}")
                    embeddings_dim = len(embeddings[0])
                    # chunked_texts = [emb['text'] for emb in embeddings]
                    # embeddings = [emb['vector'] for emb in embeddings]
                    # if len(embeddings) >= 7209543:
                    #     embeddings = embeddings[:7209543]
                    #     chunked_texts = chunked_texts[:7209543]

                # insertion
                if config["rag"]["action"]["insert"]:
                    log_time_breakdown("insert")
                    print(
                        f"***Start inserting embeddings into collection: {collection_name}"
                    )
                    if config["sys"]["vector_db"]["type"] == "lancedb":
                        db_client.create_collection(
                            collection_name=collection_name, dim=embeddings_dim
                        )
                    db_client.insert_data_vector(
                        vector=embeddings,
                        chunks=chunked_texts,
                        collection_name=collection_name,
                        insert_batch_size=config["rag"]["insert"]["batch_size"],
                        create_collection=True,
                    )
                    print(
                        f"***Insertion done, total {len(embeddings)} embeddings inserted"
                    )

                # build index
                if config["rag"]["action"]["build_index"]:
                    log_time_breakdown("build")
                    db_client.build_index(
                        collection_name=collection_name,
                        index_type=config["rag"]["build_index"]["index_type"],
                        metric_type=config["rag"]["build_index"]["metric_type"],
                        # device=None,
                        # device=device
                    )
                    print(f"***Indexing done for collection: {collection_name}")
                log_time_breakdown("done")
        # query + retrieval + reranking + generation + evaluation
        if config["rag"]["action"]["generation"] == True:
            RAGRequest = WikipediaRequests(
                run_name=config["run_name"],
                collection_name=collection_name,
                req_type="query",
                req_count=config["rag"]["retrieval"]["question_num"],
            )
            print(f"***End request preparation")

            # prepare pipeline
            from RAGPipeline.TextsRAGPipline import TextsRAGPipeline
            from RAGPipeline.responser.TextsResponser import VLLMResponser

            retriever = BaseRetriever(
                collection_name=collection_name,
                top_k=config["rag"]["retrieval"]["top_k"],
                retrieval_batch_size=config["rag"]["retrieval"]["retrieval_batch_size"],
                client=db_client,
            )
            if config["rag"]["action"]["reranking"]:
                reranker = CrossEncoderReranker(
                    model_name=config["rag"]["reranking"]["rerank_model"],
                    top_n=config["rag"]["reranking"]["top_n"],
                    device=config["rag"]["reranking"]["device"],
                )
            else:
                reranker = None
            if config["rag"]["action"]["evaluate"]:
                from evaluator.Ragasvllm import Ragasvllm

                evaluator = Ragasvllm(
                    llm_path=config["rag"]["evaluate"]["evaluator_model"],
                )
            else:
                evaluator = None
            responser = VLLMResponser(
                model=config["rag"]["generation"]["model"],
                device=config["rag"]["generation"]["device"],
                parallelism=config["rag"]["generation"]["parallelism"],
            )
            embedder = SentenceTransformerEncoder(
                device=config["rag"]["embedding"]["device"],
                sentence_transformers_name=config["rag"]["embedding"][
                    "sentence_transformers_name"
                ],
            )
            RAGPipline = TextsRAGPipeline(
                retriever=retriever,
                responser=responser,
                embedder=embedder,
                reranker=reranker,
                evaluator=evaluator,
            )

            # pipeline.check()
            import utils.colored_print as cprint

            with monitor:
                RAGPipline.process(
                    RAGRequest,
                    batch_size=config["rag"]["pipeline"]["batch_size"],
                )
        if hasattr(db_client, "close_trace"):
            db_client.close_trace()
        monitor.close()


if __name__ == "__main__":
    main()
