-- ============================================================================
-- 005 · Fuente de la curva científica en el tablero de la matriz
--
-- matrix_runs registraba la fuente de patentes (epo / google) pero no la de
-- papers: se asumía Scopus. Para comparar tecnologías medidas con APIs
-- distintas (OpenAlex, Semantic Scholar, Lens), la fuente tiene que quedar
-- guardada. Las filas existentes son todas de Scopus. Idempotente.
-- ============================================================================

begin;

alter table matrix_runs
  add column if not exists papers_source text not null default 'scopus';

alter table matrix_runs drop constraint if exists matrix_runs_papers_source_check;
alter table matrix_runs
  add constraint matrix_runs_papers_source_check
  check (papers_source in ('scopus', 'openalex', 'semantic_scholar', 'lens'));

commit;
